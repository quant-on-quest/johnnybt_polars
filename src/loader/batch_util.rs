//! RecordBatch schema 对齐工具。
//!
//! `arrow::compute::concat_batches` 按列序号无条件取列（arrow-select/src/concat.rs），
//! 一旦某个 batch 的列数少于基准 schema 就会 panic（RecordBatch::column 越界）。
//! panic 经 pyo3 变成继承 BaseException 的 PanicException，Python 侧的
//! `except Exception` 兜不住。所以合并前一律先按列名对齐。

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{new_null_array, Array, ArrayRef};
use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;

/// 取所有 batch 的列并集，保持首次出现的顺序
pub fn union_schema(batches: &[RecordBatch]) -> SchemaRef {
    let mut seen: HashSet<String> = HashSet::new();
    let mut fields: Vec<Field> = Vec::new();
    for batch in batches {
        for field in batch.schema().fields() {
            if seen.insert(field.name().clone()) {
                fields.push(field.as_ref().clone());
            }
        }
    }
    Arc::new(Schema::new(fields))
}

/// 按列名把 batch 对齐到目标 schema：缺失列补 null，类型不同尝试 cast
pub fn align_batch_to_schema(
    batch: &RecordBatch,
    target_schema: &SchemaRef,
) -> Result<RecordBatch, String> {
    // 列名与顺序完全一致时无需重建
    if batch.schema().fields() == target_schema.fields() {
        return Ok(batch.clone());
    }

    let num_rows = batch.num_rows();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(target_schema.fields().len());

    for field in target_schema.fields() {
        match batch.schema().column_with_name(field.name()) {
            Some((idx, _)) => {
                let col = batch.column(idx);
                if col.data_type() == field.data_type() {
                    columns.push(col.clone());
                } else {
                    match arrow::compute::cast(col.as_ref(), field.data_type()) {
                        Ok(casted) => columns.push(casted),
                        Err(_) => columns.push(new_null_array(field.data_type(), num_rows)),
                    }
                }
            }
            None => columns.push(new_null_array(field.data_type(), num_rows)),
        }
    }

    let opts = arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(num_rows));
    RecordBatch::try_new_with_options(target_schema.clone(), columns, &opts)
        .map_err(|e| format!("对齐 RecordBatch 失败: {e}"))
}

/// 按列名把一批 batch 对齐到同一个 schema（不做拷贝合并）
pub fn align_all(batches: &[RecordBatch]) -> Result<(SchemaRef, Vec<RecordBatch>), String> {
    if batches.is_empty() {
        return Err("无有效数据".to_string());
    }
    let schema = union_schema(batches);
    let aligned: Vec<RecordBatch> = batches
        .iter()
        .map(|b| align_batch_to_schema(b, &schema))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((schema, aligned))
}

/// 先按列名对齐再合并，避免 concat_batches 在列数不齐时 panic
pub fn concat_aligned(batches: &[RecordBatch]) -> Result<RecordBatch, String> {
    let (schema, aligned) = align_all(batches)?;
    arrow::compute::concat_batches(&schema, &aligned)
        .map_err(|e| format!("合并 RecordBatch 失败: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::DataType;

    fn batch(cols: &[(&str, DataType)], n: usize) -> RecordBatch {
        let fields: Vec<Field> = cols
            .iter()
            .map(|(name, dt)| Field::new(*name, dt.clone(), true))
            .collect();
        let arrays: Vec<ArrayRef> = cols
            .iter()
            .map(|(_, dt)| match dt {
                DataType::Utf8 => {
                    Arc::new(StringArray::from(vec!["x"; n])) as ArrayRef
                }
                _ => Arc::new(Float64Array::from(vec![1.0; n])) as ArrayRef,
            })
            .collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
    }

    #[test]
    fn concat_with_missing_columns_does_not_panic() {
        // issue #1 案例 3：列数不齐，旧实现在这里 panic
        let a = batch(&[("x", DataType::Float64), ("y", DataType::Float64)], 2);
        let b = batch(&[("x", DataType::Float64)], 3);
        let out = concat_aligned(&[a, b]).unwrap();
        assert_eq!(out.num_rows(), 5);
        assert_eq!(out.num_columns(), 2);
        assert_eq!(out.column(1).null_count(), 3);
    }

    #[test]
    fn concat_with_extra_columns_takes_union() {
        let a = batch(&[("x", DataType::Float64)], 1);
        let b = batch(&[("x", DataType::Float64), ("z", DataType::Utf8)], 1);
        let out = concat_aligned(&[a, b]).unwrap();
        assert_eq!(out.num_columns(), 2);
        assert_eq!(out.schema().field(1).name(), "z");
        assert_eq!(out.num_rows(), 2);
    }

    #[test]
    fn concat_with_reordered_columns_matches_by_name() {
        let a = batch(&[("x", DataType::Float64), ("y", DataType::Utf8)], 1);
        let b = batch(&[("y", DataType::Utf8), ("x", DataType::Float64)], 1);
        let out = concat_aligned(&[a, b]).unwrap();
        assert_eq!(out.schema().field(0).name(), "x");
        assert_eq!(out.schema().field(1).name(), "y");
        assert_eq!(out.column(1).as_any().downcast_ref::<StringArray>().unwrap().value(1), "x");
    }
}
