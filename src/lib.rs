use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

mod expressions;
mod io;
mod loader;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

/// The importable half: the GBK readers. The expression plugins are found
/// through the shared library's symbols and need no module at all.
#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(io::read_gbk_csvs, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_gbk_csvs_diagonal, m)?)?;
    Ok(())
}
