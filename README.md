# johnnybt-polars

The hot paths of [johnnybt](https://github.com/quant-on-quest/johnnybt) in Rust,
as polars plugins — so the parallelism lives where the work is, not behind
Python's GIL.

Four things, each because the Python version was the bottleneck:

| | what it does |
|---|---|
| `scan_gbk_csv` | Scans a directory of **GB18030** CSVs as a `LazyFrame`, decoding in Rust. An IO plugin: projection and predicates push down, files stream in batches. Measured 4.8× a polars-plus-Python decode. |
| `scatter` | Writes a long frame into dense `(rows, cols)` float64 matrices, one per column, in parallel on rayon — numpy fancy-index semantics without leaving Rust. |
| `vintage` | Expands one instrument's filings into **vintages** in a single pass: for every distinct publish date, the newest period then known and each line's value at that period and at the quarters before it. Point-in-time financials without a self-join. |
| `simulate` | Walks one account per group — a day-by-day account state machine ([`johnnybt_engine`](https://github.com/quant-on-quest/johnnybt_engine)) under its market-neutral policy: buys fill from cash the account has, blocked orders lock their exact cost, nobody runs a debt. |

Every one is an expression or IO plugin: you call it from polars, polars decides
the parallelism, and no data crosses back into Python in between.

## Install

```sh
pip install johnnybt-polars
```

Prebuilt wheels — no Rust toolchain needed.

## Use

```python
import polars as pl
from johnnybt_polars import scan_gbk_csv, vintage

frame = scan_gbk_csv(paths, schema={"股票代码": "str", "收盘价": "float64"}, skip_rows=1)
seen = frame.group_by("股票代码").agg(vintage("报告期", "公告日", "净利润", horizon_days=1))
```

## Build from source

```sh
maturin develop --release
```

Rust ≥ 1.85, Python ≥ 3.11 (abi3 — one wheel covers 3.11 through 3.14).

## Licence

MIT
