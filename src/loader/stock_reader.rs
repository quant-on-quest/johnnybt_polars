use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::loader::batch_util::align_all;
use crate::loader::chunked_io::read_parse_chunked;
use crate::loader::csv_scan::Scanner;
use crate::loader::gbk::decode_gbk;
use arrow::array::*;
use arrow::datatypes::{DataType, Date32Type, Field, Float64Type, Int64Type, Schema};
use arrow::record_batch::RecordBatch;

/// 列类型枚举，由 Python 端指定
#[derive(Clone, Debug)]
pub enum ColType {
    Str,
    Float64,
    Int64,
    Date { format: String },
}

/// 从 Python 传入的 schema 定义
pub struct SchemaSpec {
    pub string_cols: HashSet<String>,
    pub date_cols: HashMap<String, String>,
    pub int_cols: HashSet<String>,
    pub float_cols: HashSet<String>,
    /// 未在上面任何一处声明的列走这个类型
    pub default_type: ColType,
}

impl SchemaSpec {
    /// 返回 (列类型, 是否被显式声明)
    pub fn col_type(&self, name: &str) -> (ColType, bool) {
        if self.string_cols.contains(name) {
            (ColType::Str, true)
        } else if self.int_cols.contains(name) {
            (ColType::Int64, true)
        } else if self.float_cols.contains(name) {
            (ColType::Float64, true)
        } else if let Some(fmt) = self.date_cols.get(name) {
            (
                ColType::Date {
                    format: fmt.clone(),
                },
                true,
            )
        } else {
            (self.default_type.clone(), false)
        }
    }
}

/// 解析行为选项
#[derive(Clone, Debug)]
pub struct ParseOptions {
    /// 跳过文件开头的行数
    pub skip_rows: usize,
    /// 是否处理双引号
    pub quoting: bool,
    /// 是否剥离字符串字段的首尾空白（数值/日期列始终按 trim 后的值解析）
    pub trim: bool,
}

impl Default for ParseOptions {
    fn default() -> Self {
        ParseOptions {
            skip_rows: 1,
            quoting: true,
            trim: true,
        }
    }
}

// ─── 列式 Builder ────────────────────────────────────────────────

enum BuilderKind {
    Str(StringBuilder),
    F64(PrimitiveBuilder<Float64Type>),
    I64(PrimitiveBuilder<Int64Type>),
    Date(PrimitiveBuilder<Date32Type>, DateFormat),
}

pub struct ColumnBuilder {
    kind: BuilderKind,
    /// 未声明的数值列要统计解析失败，声明过的不统计（尊重调用方的选择）
    track_failures: bool,
    n_values: usize,
    n_failed: usize,
    first_bad: Option<String>,
}

/// 预解析日期格式，避免重复匹配
pub enum DateFormat {
    /// %Y-%m-%d  (YYYY-MM-DD, 10 chars)
    Ymd,
    /// %Y%m%d    (YYYYMMDD, 8 chars)
    Ymd8,
    /// 其他格式回退 chrono
    Other(String),
}

impl DateFormat {
    fn from_str(fmt: &str) -> Self {
        match fmt {
            "%Y-%m-%d" => DateFormat::Ymd,
            "%Y%m%d" => DateFormat::Ymd8,
            _ => DateFormat::Other(fmt.to_string()),
        }
    }
}

/// epoch 常量 (2000-03-01 的 days since 1970-01-01 = 11017)
/// 使用算法直接计算 days since epoch，不依赖 chrono
const EPOCH_OFFSET: i32 = 719_468; // days from 0000-03-01 to 1970-01-01

/// 快速日期解析：直接计算 days since unix epoch
#[inline]
fn fast_parse_date(s: &str, fmt: &DateFormat) -> Option<i32> {
    match fmt {
        DateFormat::Ymd => {
            // "2024-01-15" → 10 chars
            let b = s.as_bytes();
            if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
                return None;
            }
            let y = parse_digits::<4>(b, 0)? as i32;
            let m = parse_digits::<2>(b, 5)?;
            let d = parse_digits::<2>(b, 8)?;
            civil_to_days(y, m, d)
        }
        DateFormat::Ymd8 => {
            // "20240115" → 8 chars
            let b = s.as_bytes();
            if b.len() != 8 {
                return None;
            }
            let y = parse_digits::<4>(b, 0)? as i32;
            let m = parse_digits::<2>(b, 4)?;
            let d = parse_digits::<2>(b, 6)?;
            civil_to_days(y, m, d)
        }
        DateFormat::Other(fmt_str) => {
            use chrono::NaiveDate;
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1)?;
            NaiveDate::parse_from_str(s, fmt_str)
                .ok()
                .map(|d| (d - epoch).num_days() as i32)
        }
    }
}

/// 从字节切片解析 N 位数字
#[inline]
fn parse_digits<const N: usize>(b: &[u8], offset: usize) -> Option<u32> {
    let mut val: u32 = 0;
    for i in 0..N {
        let c = b[offset + i];
        if !c.is_ascii_digit() {
            return None;
        }
        val = val * 10 + (c - b'0') as u32;
    }
    Some(val)
}

/// Civil date → days since Unix epoch (1970-01-01)
/// 算法来自 Howard Hinnant: http://howardhinnant.github.io/date_algorithms.html
#[inline]
fn civil_to_days(y: i32, m: u32, d: u32) -> Option<i32> {
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400) as u32;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe as i32 - EPOCH_OFFSET;
    Some(days)
}

impl ColumnBuilder {
    pub fn new(col_type: &ColType, declared: bool) -> Self {
        let kind = match col_type {
            ColType::Str => BuilderKind::Str(StringBuilder::new()),
            ColType::Float64 => BuilderKind::F64(PrimitiveBuilder::<Float64Type>::new()),
            ColType::Int64 => BuilderKind::I64(PrimitiveBuilder::<Int64Type>::new()),
            ColType::Date { format } => BuilderKind::Date(
                PrimitiveBuilder::<Date32Type>::new(),
                DateFormat::from_str(format),
            ),
        };
        let track_failures = !declared && matches!(kind, BuilderKind::F64(_) | BuilderKind::I64(_));
        ColumnBuilder {
            kind,
            track_failures,
            n_values: 0,
            n_failed: 0,
            first_bad: None,
        }
    }

    #[inline]
    pub fn append(&mut self, val: &str) {
        match &mut self.kind {
            BuilderKind::Str(b) => {
                if val.is_empty() {
                    b.append_null();
                } else {
                    b.append_value(val);
                }
            }
            BuilderKind::F64(b) => {
                // 数值列始终按 trim 后的值解析，与 trim 选项无关
                let val = val.trim();
                if val.is_empty() {
                    b.append_null();
                } else {
                    match val.parse::<f64>() {
                        Ok(v) => b.append_value(v),
                        Err(_) => {
                            b.append_null();
                            if self.track_failures {
                                self.n_failed += 1;
                                if self.first_bad.is_none() {
                                    self.first_bad = Some(val.to_string());
                                }
                            }
                        }
                    }
                    if self.track_failures {
                        self.n_values += 1;
                    }
                }
            }
            BuilderKind::I64(b) => {
                let val = val.trim();
                if val.is_empty() {
                    b.append_null();
                } else {
                    match val.parse::<i64>() {
                        Ok(v) => b.append_value(v),
                        Err(_) => {
                            b.append_null();
                            if self.track_failures {
                                self.n_failed += 1;
                                if self.first_bad.is_none() {
                                    self.first_bad = Some(val.to_string());
                                }
                            }
                        }
                    }
                    if self.track_failures {
                        self.n_values += 1;
                    }
                }
            }
            BuilderKind::Date(b, fmt) => {
                let val = val.trim();
                if val.is_empty() {
                    b.append_null();
                } else {
                    match fast_parse_date(val, fmt) {
                        Some(days) => b.append_value(days),
                        None => b.append_null(),
                    }
                }
            }
        }
    }

    /// 未声明的数值列如果非空值全军覆没，八成是调用方漏声明了字符串列。
    /// 这种情况以前静默产出全 null，现在报错。
    pub fn check_all_failed(&self, name: &str) -> Result<(), String> {
        if self.track_failures && self.n_values > 0 && self.n_failed == self.n_values {
            let type_name = match self.kind {
                BuilderKind::I64(_) => "int64",
                _ => "float64",
            };
            let sample = self.first_bad.as_deref().unwrap_or("");
            return Err(format!(
                "列 '{name}' 未在 schema 中声明，按 {type_name} 解析，\
                 但 {} 个非空值全部解析失败（示例值: '{sample}'）。\
                 请在 schema 里把它声明为 \"str\"，或调用时传 default_type=\"str\"",
                self.n_values
            ));
        }
        Ok(())
    }

    pub fn finish(self, name: &str) -> (Field, Arc<dyn Array>) {
        match self.kind {
            BuilderKind::Str(mut b) => {
                let arr = b.finish();
                (Field::new(name, DataType::Utf8, true), Arc::new(arr))
            }
            BuilderKind::F64(mut b) => {
                let arr = b.finish();
                (Field::new(name, DataType::Float64, true), Arc::new(arr))
            }
            BuilderKind::I64(mut b) => {
                let arr = b.finish();
                (Field::new(name, DataType::Int64, true), Arc::new(arr))
            }
            BuilderKind::Date(mut b, _) => {
                let arr = b.finish();
                (Field::new(name, DataType::Date32, true), Arc::new(arr))
            }
        }
    }
}

// ─── 从内存字节解析 CSV → RecordBatch ────────────────────────

/// 从已读取的字节解析 CSV（不做 I/O，纯 CPU）
pub fn parse_csv_from_bytes(
    raw: &[u8],
    columns: Option<&[String]>,
    schema_spec: &SchemaSpec,
    opts: &ParseOptions,
) -> Result<RecordBatch, String> {
    let text = decode_gbk(raw);
    parse_csv_from_text(&text, columns, schema_spec, opts)
}

/// 解析已解码为 UTF-8 的 CSV 文本
///
/// `columns` 按**列名**映射，顺序任意；输出列顺序与 `columns` 一致。
/// 文件里没有的列输出为整列 null，而不是被悄悄丢掉。
pub fn parse_csv_from_text(
    text: &str,
    columns: Option<&[String]>,
    schema_spec: &SchemaSpec,
    opts: &ParseOptions,
) -> Result<RecordBatch, String> {
    let mut scanner = Scanner::new(text, opts.skip_rows, opts.quoting, opts.trim);

    // 表头（表头始终 trim，否则列名匹配不上）
    let all_headers = scanner.read_row().ok_or_else(|| "文件无表头".to_string())?;

    // 输出列名（顺序 = 调用方要求的顺序）
    let out_names: Vec<&str> = match columns {
        Some(cols) => cols.iter().map(|c| c.as_str()).collect(),
        None => all_headers.iter().map(|h| h.as_str()).collect(),
    };

    // 输出列 → 文件里的字段下标；scan_order 按文件列序排好，扫描才能一遍过
    let mut scan_order: Vec<(usize, usize)> = Vec::with_capacity(out_names.len()); // (file_idx, out_pos)
    let mut absent: Vec<usize> = Vec::new(); // 文件里没有的输出列
    match columns {
        Some(cols) => {
            let idx_map: HashMap<&str, usize> = all_headers
                .iter()
                .enumerate()
                .map(|(i, h)| (h.as_str(), i))
                .collect();
            for (out_pos, col) in cols.iter().enumerate() {
                match idx_map.get(col.as_str()) {
                    Some(&file_idx) => scan_order.push((file_idx, out_pos)),
                    None => absent.push(out_pos),
                }
            }
            scan_order.sort_unstable_by_key(|(file_idx, _)| *file_idx);
        }
        None => {
            for i in 0..all_headers.len() {
                scan_order.push((i, i));
            }
        }
    }

    // 创建列式 builders（按输出顺序）
    let mut builders: Vec<ColumnBuilder> = out_names
        .iter()
        .map(|name| {
            let (col_type, declared) = schema_spec.col_type(name);
            ColumnBuilder::new(&col_type, declared)
        })
        .collect();

    // 流式解析：逐条记录扫描，直接 append 到 builder（普通字段零拷贝）
    let mut cursor;
    loop {
        cursor = 0;
        let got = scanner.next_record(|field_idx, val| {
            if cursor < scan_order.len() && field_idx == scan_order[cursor].0 {
                builders[scan_order[cursor].1].append(val);
                cursor += 1;
            }
            // 需要的列都读完了就让扫描器快进到行尾
            cursor < scan_order.len()
        });
        if !got {
            break;
        }
        // 这一行没扫到的列补 null
        for (_, out_pos) in &scan_order[cursor..] {
            builders[*out_pos].append("");
        }
        for out_pos in &absent {
            builders[*out_pos].append("");
        }
    }

    // builders → RecordBatch
    let mut fields = Vec::with_capacity(out_names.len());
    let mut arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(out_names.len());
    for (name, builder) in out_names.into_iter().zip(builders) {
        builder.check_all_failed(name)?;
        let (field, array) = builder.finish(name);
        fields.push(field);
        arrays.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays).map_err(|e| format!("构建 RecordBatch 失败: {e}"))
}

// ─── 批量读取：分块 I/O + rayon CPU 解析 ─────────────────────────

/// 将多个 CSV 文件并行读取，合并为按列名对齐的 RecordBatch 列表
pub fn read_csvs_to_batches(
    paths: &[String],
    columns: Option<&[String]>,
    schema_spec: &SchemaSpec,
    opts: &ParseOptions,
    io_threads: usize,
) -> Result<(arrow::datatypes::SchemaRef, Vec<RecordBatch>), String> {
    let batches = read_parse_chunked(paths, io_threads, |bytes, path| {
        parse_csv_from_bytes(bytes, columns, schema_spec, opts).map_err(|e| format!("{path}: {e}"))
    })?;
    align_all(&batches)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用 schema：其余列走 default_type
    fn spec(strings: &[&str], ints: &[&str], default_type: ColType) -> SchemaSpec {
        SchemaSpec {
            string_cols: strings.iter().map(|s| s.to_string()).collect(),
            date_cols: HashMap::new(),
            int_cols: ints.iter().map(|s| s.to_string()).collect(),
            float_cols: HashSet::new(),
            default_type,
        }
    }

    fn opts() -> ParseOptions {
        ParseOptions {
            skip_rows: 1,
            quoting: true,
            trim: true,
        }
    }

    fn cols(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn strs(batch: &RecordBatch, name: &str) -> Vec<Option<String>> {
        let (idx, _) = batch.schema().column_with_name(name).expect("列不存在");
        let a = batch
            .column(idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("不是字符串列");
        (0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                }
            })
            .collect()
    }

    #[test]
    fn columns_are_mapped_by_name_not_position() {
        // 文件列序 C,A,B；请求 [A,B,C]
        let text = "免责行\nC,A,B\nC0,A0,B0\n";
        let want = cols(&["A", "B", "C"]);
        let batch = parse_csv_from_text(
            text,
            Some(&want),
            &spec(&["A", "B", "C"], &[], ColType::Float64),
            &opts(),
        )
        .unwrap();

        assert_eq!(
            batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>(),
            vec!["A", "B", "C"]
        );
        assert_eq!(strs(&batch, "C"), vec![Some("C0".into())]);
        assert_eq!(strs(&batch, "A"), vec![Some("A0".into())]);
    }

    #[test]
    fn requested_column_absent_from_file_becomes_null_column() {
        let text = "免责行\nA,B\na0,b0\n";
        let want = cols(&["A", "缺失", "B"]);
        let batch = parse_csv_from_text(
            text,
            Some(&want),
            &spec(&["A", "B", "缺失"], &[], ColType::Float64),
            &opts(),
        )
        .unwrap();

        assert_eq!(batch.num_columns(), 3);
        assert_eq!(strs(&batch, "缺失"), vec![None]);
    }

    #[test]
    fn undeclared_column_that_never_parses_is_an_error() {
        let text = "免责行\n代码,标题\nsh600000,某公告\n";
        let err = parse_csv_from_text(text, None, &spec(&["代码"], &[], ColType::Float64), &opts())
            .unwrap_err();

        assert!(err.contains("标题"), "错误信息要指出列名: {err}");
        assert!(err.contains("某公告"), "错误信息要给出示例值: {err}");
    }

    #[test]
    fn declared_float_column_that_never_parses_is_not_an_error() {
        let text = "免责行\nv\n--\n";
        let mut s = spec(&[], &[], ColType::Float64);
        s.float_cols.insert("v".to_string());
        let batch = parse_csv_from_text(text, None, &s, &opts()).unwrap();

        assert_eq!(batch.column(0).null_count(), 1);
    }

    #[test]
    fn default_type_str_keeps_undeclared_columns_as_text() {
        let text = "免责行\n代码,标题\nsh600000,某公告\n";
        let batch =
            parse_csv_from_text(text, None, &spec(&[], &[], ColType::Str), &opts()).unwrap();

        assert_eq!(strs(&batch, "标题"), vec![Some("某公告".into())]);
    }

    #[test]
    fn int64_keeps_precision_beyond_float64() {
        let text = "免责行\na,v\nx,9007199254740993\nx,-42\nx,\nx,不是数字\nx,1.5\n";
        let batch =
            parse_csv_from_text(text, None, &spec(&["a"], &["v"], ColType::Float64), &opts())
                .unwrap();

        let (i, _) = batch.schema().column_with_name("v").unwrap();
        let a = batch
            .column(i)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(a.value(0), 9007199254740993);
        assert_eq!(a.value(1), -42);
        assert!(a.is_null(2), "空值 → null");
        assert!(a.is_null(3), "非数字 → null");
        assert!(a.is_null(4), "小数 → null");
    }

    #[test]
    fn int64_overflow_becomes_null() {
        let text = "免责行\nv\n99999999999999999999999\n";
        let batch =
            parse_csv_from_text(text, None, &spec(&[], &["v"], ColType::Float64), &opts()).unwrap();

        assert_eq!(batch.column(0).null_count(), 1);
    }

    #[test]
    fn trim_false_preserves_whitespace_in_both_quoted_and_plain_fields() {
        let text = "免责行\na,b\n  空格  ,\"  引号内  \"\n";
        let mut o = opts();
        o.trim = false;
        let batch =
            parse_csv_from_text(text, None, &spec(&["a", "b"], &[], ColType::Float64), &o).unwrap();

        assert_eq!(strs(&batch, "a"), vec![Some("  空格  ".into())]);
        assert_eq!(strs(&batch, "b"), vec![Some("  引号内  ".into())]);
    }

    #[test]
    fn trim_true_also_trims_quoted_fields() {
        let text = "免责行\na,b\n  空格  ,\"  引号内  \"\n";
        let batch = parse_csv_from_text(
            text,
            None,
            &spec(&["a", "b"], &[], ColType::Float64),
            &opts(),
        )
        .unwrap();

        assert_eq!(strs(&batch, "a"), vec![Some("空格".into())]);
        assert_eq!(strs(&batch, "b"), vec![Some("引号内".into())]);
    }

    #[test]
    fn numbers_parse_even_when_trim_is_off() {
        let text = "免责行\nv\n  1.5  \n";
        let mut o = opts();
        o.trim = false;
        let batch = parse_csv_from_text(text, None, &spec(&[], &[], ColType::Float64), &o).unwrap();

        let a = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(a.value(0), 1.5);
    }

    #[test]
    fn headers_are_trimmed_even_when_trim_is_off() {
        let text = "免责行\n a , b \nx,y\n";
        let mut o = opts();
        o.trim = false;
        let want = cols(&["a", "b"]);
        let batch = parse_csv_from_text(
            text,
            Some(&want),
            &spec(&["a", "b"], &[], ColType::Float64),
            &o,
        )
        .unwrap();

        assert_eq!(strs(&batch, "a"), vec![Some("x".into())]);
    }
}
