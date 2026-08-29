//! Reading the vendor's GB18030 CSVs into polars, wholly in Rust.
//!
//! The bridge the standalone loader used went arrow → IPC bytes → pyarrow →
//! `pl.from_arrow` — two Python libraries in the middle of a hot path. Here
//! the IPC stream is read back by polars itself, in Rust, and the DataFrame
//! crosses into Python once.

use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::panic::{catch_unwind, AssertUnwindSafe};

use arrow::ipc::writer::StreamWriter;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_polars::PyDataFrame;

use crate::loader::stock_reader::{ColType, ParseOptions, SchemaSpec};
use crate::loader::{fina_reader, stock_reader};

/// Parse "str" / "int64" / "float64" / "date:FMT" into a column type.
fn parse_col_type(typ: &str) -> PyResult<ColType> {
    match typ {
        "str" => Ok(ColType::Str),
        "int64" => Ok(ColType::Int64),
        "float64" => Ok(ColType::Float64),
        other => match other.strip_prefix("date:") {
            Some(fmt) => Ok(ColType::Date {
                format: fmt.to_string(),
            }),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown column type {other:?}; use \"str\" / \"int64\" / \"float64\" / \"date:%Y-%m-%d\""
            ))),
        },
    }
}

fn parse_schema(overrides: Option<&Bound<PyDict>>, default_type: &str) -> PyResult<SchemaSpec> {
    let mut string_cols = HashSet::new();
    let mut date_cols = HashMap::new();
    let mut int_cols = HashSet::new();
    let mut float_cols = HashSet::new();

    if let Some(d) = overrides {
        for (key, val) in d.iter() {
            let col: String = key.extract()?;
            let typ: String = val.extract()?;
            match parse_col_type(&typ)? {
                ColType::Str => {
                    string_cols.insert(col);
                }
                ColType::Int64 => {
                    int_cols.insert(col);
                }
                ColType::Float64 => {
                    float_cols.insert(col);
                }
                ColType::Date { format } => {
                    date_cols.insert(col, format);
                }
            }
        }
    }

    Ok(SchemaSpec {
        string_cols,
        date_cols,
        int_cols,
        float_cols,
        default_type: parse_col_type(default_type)?,
    })
}

/// Convert a panic into an ordinary RuntimeError rather than a
/// BaseException that `except Exception` cannot catch.
fn guard<T>(f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(e) => {
            let msg = e
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| e.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown".to_string());
            Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "internal error while reading CSVs: {msg}"
            )))
        }
    }
}

/// Arrow batches → in-memory IPC → polars, without leaving Rust.
fn batches_to_frame(
    schema: arrow::datatypes::SchemaRef,
    batches: Vec<arrow::record_batch::RecordBatch>,
) -> PyResult<PyDataFrame> {
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &schema)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("IPC writer: {e}")))?;
        for batch in &batches {
            writer
                .write(batch)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("IPC write: {e}")))?;
        }
        writer
            .finish()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("IPC finish: {e}")))?;
    }
    drop(batches);

    let frame = polars::prelude::IpcStreamReader::new(Cursor::new(buf))
        .finish()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("IPC read: {e}")))?;
    Ok(PyDataFrame(frame))
}

#[pyfunction]
#[pyo3(signature = (paths, columns=None, skip_rows=1, schema=None, io_threads=256, quoting=true, default_type="float64", trim=true))]
#[allow(clippy::too_many_arguments)]
pub fn read_gbk_csvs(
    paths: Vec<String>,
    columns: Option<Vec<String>>,
    skip_rows: usize,
    schema: Option<&Bound<PyDict>>,
    io_threads: usize,
    quoting: bool,
    default_type: &str,
    trim: bool,
) -> PyResult<PyDataFrame> {
    let schema_spec = parse_schema(schema, default_type)?;
    let opts = ParseOptions {
        skip_rows,
        quoting,
        trim,
    };
    let (schema, batches) = guard(|| {
        stock_reader::read_csvs_to_batches(&paths, columns.as_deref(), &schema_spec, &opts, io_threads)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    })?;
    batches_to_frame(schema, batches)
}

#[pyfunction]
#[pyo3(signature = (paths, skip_rows=1, schema=None, io_threads=256, quoting=true, default_type="float64", trim=true))]
#[allow(clippy::too_many_arguments)]
pub fn read_gbk_csvs_diagonal(
    paths: Vec<String>,
    skip_rows: usize,
    schema: Option<&Bound<PyDict>>,
    io_threads: usize,
    quoting: bool,
    default_type: &str,
    trim: bool,
) -> PyResult<PyDataFrame> {
    let schema_spec = parse_schema(schema, default_type)?;
    let opts = ParseOptions {
        skip_rows,
        quoting,
        trim,
    };
    let renames: HashMap<String, String> = HashMap::new();
    let (schema, batches) = guard(|| {
        fina_reader::read_fina_csvs_to_batches(&paths, &schema_spec, &renames, &opts, io_threads)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    })?;
    batches_to_frame(schema, batches)
}
