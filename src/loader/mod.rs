//! The GBK CSV loader, absorbed from the standalone d2-loader crate.
//!
//! One decoder, one home: the vendor's GB18030 files are read here — Rust
//! decode, Rust column building — and cross into Python exactly once, as a
//! polars DataFrame. The former `gbk-csv-loader` PyPI dependency is gone.

pub mod batch_util;
pub mod chunked_io;
pub mod csv_scan;
pub mod fina_reader;
pub mod gbk;
pub mod stock_reader;
