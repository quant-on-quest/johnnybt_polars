use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;

use crate::loader::batch_util::align_all;
use crate::loader::chunked_io::read_parse_chunked;
use crate::loader::csv_scan::Scanner;
use crate::loader::gbk::decode_gbk;
use crate::loader::stock_reader::{ColType, ColumnBuilder, ParseOptions, SchemaSpec};

/// 排除的列前缀
const EXCLUDE_PREFIXES: &[&str] = &["Unnamed", "抓取时间"];

/// 从字节解析单个财务 CSV → RecordBatch
fn parse_fina_csv_from_bytes(
    raw: &[u8],
    schema_spec: &SchemaSpec,
    renames: &HashMap<String, String>,
    opts: &ParseOptions,
) -> Result<RecordBatch, String> {
    let text = decode_gbk(raw);
    let mut scanner = Scanner::new(&text, opts.skip_rows, opts.quoting, opts.trim);

    let raw_headers = scanner.read_row().ok_or_else(|| "文件无表头".to_string())?;

    // 过滤排除列，确定保留的列索引和最终列名
    let mut keep_indices: Vec<usize> = Vec::new();
    let mut final_headers: Vec<String> = Vec::new();
    for (i, h) in raw_headers.iter().enumerate() {
        if h.is_empty() || EXCLUDE_PREFIXES.iter().any(|prefix| h.starts_with(prefix)) {
            continue;
        }
        keep_indices.push(i);
        let name = renames.get(h).cloned().unwrap_or_else(|| h.to_string());
        final_headers.push(name);
    }

    // 创建 builders
    let col_types: Vec<(ColType, bool)> = final_headers
        .iter()
        .map(|h| schema_spec.col_type(h))
        .collect();
    let mut builders: Vec<ColumnBuilder> = col_types
        .iter()
        .map(|(t, declared)| ColumnBuilder::new(t, *declared))
        .collect();

    // 流式解析
    let mut cursor;
    loop {
        cursor = 0;
        let got = scanner.next_record(|field_idx, val| {
            if cursor < keep_indices.len() && field_idx == keep_indices[cursor] {
                builders[cursor].append(val);
                cursor += 1;
            }
            cursor < keep_indices.len()
        });
        if !got {
            break;
        }
        // 字段数不够的行补 null
        while cursor < builders.len() {
            builders[cursor].append("");
            cursor += 1;
        }
    }

    // builders → RecordBatch
    let mut fields = Vec::with_capacity(final_headers.len());
    let mut arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(final_headers.len());
    for (header, builder) in final_headers.iter().zip(builders) {
        builder.check_all_failed(header)?;
        let (field, array) = builder.finish(header);
        fields.push(field);
        arrays.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays).map_err(|e| format!("构建 RecordBatch 失败: {e}"))
}

/// 批量读取财务 CSV（异构 schema，diagonal concat）
/// tokio 异步 I/O + rayon CPU 解析
pub fn read_fina_csvs_to_batches(
    paths: &[String],
    schema_spec: &SchemaSpec,
    renames: &HashMap<String, String>,
    opts: &ParseOptions,
    io_threads: usize,
) -> Result<(arrow::datatypes::SchemaRef, Vec<RecordBatch>), String> {
    let batches = read_parse_chunked(paths, io_threads, |bytes, path| {
        parse_fina_csv_from_bytes(bytes, schema_spec, renames, opts)
            .map_err(|e| format!("{path}: {e}"))
    })?;

    // 列名并集 + 缺失列补 null（diagonal concat）
    align_all(&batches)
}
