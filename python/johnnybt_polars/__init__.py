"""Polars expression plugins for johnnybt.

The hot paths of panel construction, in Rust, parallel inside polars' own
engine — no Python threads, no Rust-to-Python round trips mid-computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

import polars as pl
from polars.io.plugins import register_io_source
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_LIB = Path(__file__).parent


def abi_probe(expr: pl.Expr | str) -> pl.Expr:
    """Add one to an integer column; exists to prove the plugin loads.

    Args:
        expr: The column.

    Returns:
        The expression.
    """
    return register_plugin_function(plugin_path=_LIB, args=[expr], function_name="abi_probe", is_elementwise=True)


def vintage(
    report: pl.Expr | str,
    publish: pl.Expr | str,
    *lines: str,
    published_name: str | None = None,
    horizon_days: int,
    offsets: int = 8,
) -> pl.Expr:
    """Expand one instrument's filings into vintages, in one pass.

    Use under `group_by(key).agg(...)`: polars parallelises the groups on its
    own thread pool. The output is a struct series — one row per distinct
    publish date, carrying the newest known period and each line's visible
    value at that period and at each of `offsets` quarters before it —
    explode and unnest it.

    Args:
        report: The report-date column.
        publish: The publish-date column.
        *lines: The statement lines.
        horizon_days: The frame-wide latest publish date, as days since
            epoch; unfiled-deadline rows reach up to it.
        offsets: How many quarters behind the newest period the views read.

    Returns:
        The expression.
    """
    names = [line if isinstance(line, str) else str(line) for line in lines]
    return register_plugin_function(
        plugin_path=_LIB,
        args=[report, publish, *lines],
        function_name="vintage",
        is_elementwise=False,
        kwargs={
            "horizon_days": horizon_days,
            "offsets": offsets,
            "published": published_name or (publish if isinstance(publish, str) else "publish"),
            "lines": names,
        },
    )


def scan_gbk_csv(
    paths: "Sequence[str | Path]",
    *,
    schema: dict[str, str],
    skip_rows: int = 1,
    default_type: str = "float64",
    diagonal: bool = False,
    files_per_batch: int = 256,
) -> pl.LazyFrame:
    """Scan a vendor's GB18030 CSV resource as a LazyFrame.

    An IO plugin over this crate's own Rust loader (absorbed from the former
    standalone gbk-csv-loader): projection pushes down into the decoder —
    only the selected columns are ever parsed — a predicate is applied per
    batch, and the files stream through in batches rather than landing as one
    frame. The decode never touches Python: each batch crosses the boundary
    once, already a polars DataFrame.

    Args:
        paths: The CSV files, one instrument per file.
        schema: Column name to loader type (`str`, `float64`, `int64`,
            `date:%Y-%m-%d`, ...).
        skip_rows: Lines to drop before the header.
        default_type: The type for columns the schema does not name.
        diagonal: Concatenate heterogeneous files diagonally — the financial
            product's column set depends on the accounting format — with
            missing columns as typed nulls.
        files_per_batch: How many files each yielded batch covers.

    Returns:
        A LazyFrame over the resource.
    """
    from johnnybt_polars import _lib

    named = list(schema)
    resolved = [str(path) for path in paths]

    def _dtype(kind: str) -> pl.DataType:
        if kind.startswith("date"):
            return pl.Date()
        if kind == "str":
            return pl.String()
        if kind == "int64":
            return pl.Int64()
        return pl.Float64()

    out_schema = {name: _dtype(kind) for name, kind in schema.items()}

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> "Iterator[pl.DataFrame]":
        wanted = [name for name in (with_columns or named) if name in schema]
        remaining = n_rows
        for start in range(0, len(resolved), files_per_batch):
            batch_paths = resolved[start : start + files_per_batch]
            if diagonal:
                frame = _lib.read_gbk_csvs_diagonal(
                    paths=batch_paths, skip_rows=skip_rows, schema=schema, default_type=default_type
                )
                for name in wanted:
                    if name not in frame.columns:
                        frame = frame.with_columns(pl.lit(None, dtype=out_schema[name]).alias(name))
                frame = frame.select(wanted)
            else:
                frame = _lib.read_gbk_csvs(
                    paths=batch_paths,
                    columns=wanted,
                    skip_rows=skip_rows,
                    schema={name: schema[name] for name in wanted},
                    default_type=default_type,
                )
            if predicate is not None:
                frame = frame.filter(predicate)
            if remaining is not None:
                frame = frame.head(remaining)
                remaining -= frame.height
            yield frame
            if remaining is not None and remaining <= 0:
                return

    return register_io_source(source, schema=pl.Schema(out_schema))
