//! Scattering a long frame into dense (T, N) matrices, parallel over columns.
//!
//! Every panel build ends here: each value column of the positions frame is
//! written into its own bars-by-instruments float64 matrix at `(_row, _col)`.
//! The columns are independent, so they fill on rayon's pool; numpy does the
//! same writes serially, one fancy index per column. The buffers move into
//! numpy arrays without a copy (`into_pyarray` hands over the Vec).

use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;

fn runtime_err(context: &str, e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("{context}: {e}"))
}

/// Read an Int32 index column as one contiguous slice.
fn index_column(df: &DataFrame, name: &str) -> PyResult<Vec<i32>> {
    let ca = df
        .column(name)
        .map_err(|e| runtime_err(name, e))?
        .i32()
        .map_err(|e| runtime_err(name, e))?
        .rechunk();
    match ca.cont_slice() {
        Ok(slice) => Ok(slice.to_vec()),
        Err(_) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be Int32 without nulls"
        ))),
    }
}

/// Fill one (rows, cols) NaN-initialised buffer from a value column.
///
/// Everything becomes a float so that missing has a representation, and a
/// duplicated grid position keeps the last row — the same two rules numpy's
/// fancy-index assignment follows on the fallback path.
fn fill(
    column: &Column,
    at_row: &[i32],
    at_col: &[i32],
    rows: usize,
    cols: usize,
) -> PolarsResult<Vec<f64>> {
    let values = column.cast(&DataType::Float64)?;
    let ca = values.f64()?.rechunk();
    let mut buf = vec![f64::NAN; rows * cols];
    for (i, value) in ca.iter().enumerate() {
        let r = at_row[i] as usize;
        let c = at_col[i] as usize;
        buf[r * cols + c] = value.unwrap_or(f64::NAN);
    }
    Ok(buf)
}

#[pyfunction]
pub fn scatter<'py>(
    py: Python<'py>,
    frame: PyDataFrame,
    names: Vec<String>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let df: DataFrame = frame.into();
    let at_row = index_column(&df, "_row")?;
    let at_col = index_column(&df, "_col")?;
    for (index, limit, what) in [(&at_row, rows, "_row"), (&at_col, cols, "_col")] {
        if index.iter().any(|&i| i < 0 || i as usize >= limit) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{what} holds a position outside the ({rows}, {cols}) grid"
            )));
        }
    }

    // Pure Rust from here: release the GIL so the pool actually runs.
    let buffers: Vec<(String, Vec<f64>)> = py.detach(|| {
        names
            .par_iter()
            .map(|name| {
                let column = df.column(name).map_err(|e| runtime_err(name, e))?;
                let buf =
                    fill(column, &at_row, &at_col, rows, cols).map_err(|e| runtime_err(name, e))?;
                Ok((name.clone(), buf))
            })
            .collect::<PyResult<Vec<_>>>()
    })?;

    let out = PyDict::new(py);
    for (name, buf) in buffers {
        let array = buf.into_pyarray(py);
        let matrix: Bound<'py, PyArray2<f64>> = array
            .reshape([rows, cols])
            .map_err(|e| runtime_err("reshape", e))?;
        out.set_item(name, matrix)?;
    }
    Ok(out)
}
