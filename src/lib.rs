use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

mod expressions;
mod io;
mod loader;
mod scatter;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

/// The importable half: the GBK readers and the panel scatter. The
/// expression plugins are found through the shared library's symbols and
/// need no module at all.
#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(io::read_gbk_csvs, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_gbk_csvs_diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(scatter::scatter, m)?)?;
    Ok(())
}
