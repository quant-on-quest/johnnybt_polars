use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

/// ABI smoke test: adds one to an i64 column.
#[polars_expr(output_type=Int64)]
fn abi_probe(inputs: &[Series]) -> PolarsResult<Series> {
    let s = inputs[0].i64()?;
    Ok((s + 1).into_series())
}

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct VintageKwargs {
    /// The latest publish date in the whole frame, as days since epoch. The
    /// unfiled-deadline rows reach up to it, and a group cannot know it.
    horizon_days: i32,
    /// How many quarters behind the newest known period the views read.
    offsets: usize,
    /// The publish-date field's name in the output. Passed explicitly:
    /// series names do not survive the FFI in an agg context.
    published: String,
    /// The statement lines' names, in input order.
    lines: Vec<String>,
}

/// Statutory filing deadlines: quarter-in-year -> (years later, month, day).
const DEADLINES: [(i32, u32, u32); 4] = [(0, 4, 30), (0, 8, 31), (0, 10, 31), (1, 4, 30)];

/// Days since the Unix epoch for a civil date (Howard Hinnant's algorithm).
fn days_from_civil(y: i32, m: u32, d: u32) -> i32 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as i64;
    let mp = ((m + 9) % 12) as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era as i64 * 146097 + doe - 719468) as i32
}

/// Civil (year, month) for days since the Unix epoch.
fn civil_from_days(days: i32) -> (i32, u32) {
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    ((if m <= 2 { y + 1 } else { y }) as i32, m)
}

/// `year * 4 + quarter - 1`, so subtracting four is a year.
fn quarter_index(days: i32) -> i32 {
    let (y, m) = civil_from_days(days);
    y * 4 + (m as i32 - 1) / 3
}

/// The date a quarter's report is due.
fn deadline_days(quarter: i32) -> i32 {
    let (later, month, day) = DEADLINES[(quarter.rem_euclid(4)) as usize];
    days_from_civil(quarter.div_euclid(4) + later, month, day)
}

fn vintage_fields(kwargs: &VintageKwargs) -> Vec<Field> {
    let mut fields = vec![
        Field::new(kwargs.published.as_str().into(), DataType::Date),
        Field::new("_current".into(), DataType::Int32),
    ];
    for k in 0..kwargs.offsets {
        for line in &kwargs.lines {
            let name = if k == 0 {
                line.clone()
            } else {
                format!("{line}_{k}")
            };
            fields.push(Field::new(name.into(), DataType::Float64));
        }
    }
    fields
}

fn vintage_output(_input_fields: &[Field], kwargs: VintageKwargs) -> PolarsResult<Field> {
    Ok(Field::new(
        "vintage".into(),
        DataType::Struct(vintage_fields(&kwargs)),
    ))
}

/// One instrument's filings, expanded into vintages.
///
/// Inputs: the report date, the publish date, then each statement line. The
/// output has one row per distinct publish date: what the market knew once
/// that day's filings were out — the newest known period, and each line's
/// value at that period and at each of `offsets` quarters before it, always
/// the latest version visible on or before that day. A quarter that came due
/// unfiled enters at its statutory deadline carrying nothing, so a company
/// that stopped reporting reads as missing rather than as its stale numbers.
///
/// Called under `group_by(key).agg(...)`: polars parallelises the groups on
/// its own thread pool, which is the whole reason this lives here rather than
/// as eight joins in Python.
#[polars_expr(output_type_func_with_kwargs=vintage_output)]
fn vintage(inputs: &[Series], kwargs: VintageKwargs) -> PolarsResult<Series> {
    let report = inputs[0].date()?.physical();
    let publish = inputs[1].date()?.physical();
    let lines: Vec<&Float64Chunked> = inputs[2..]
        .iter()
        .map(|s| s.f64())
        .collect::<PolarsResult<_>>()?;
    let width = lines.len();
    let offsets = kwargs.offsets;

    // (publish, quarter, values), skipping rows with no dates. The input
    // order is kept through a stable sort, so "the later row wins" below
    // means what it means in the file.
    let mut rows: Vec<(i32, i32, Vec<Option<f64>>)> = Vec::with_capacity(report.len());
    for i in 0..report.len() {
        let (Some(r), Some(p)) = (report.get(i), publish.get(i)) else {
            continue;
        };
        rows.push((
            p,
            quarter_index(r),
            lines.iter().map(|c| c.get(i)).collect(),
        ));
    }
    rows.sort_by_key(|(p, q, _)| (*p, *q));

    // One filing per (publish, quarter): the later row is the one the day
    // ended with.
    let mut dedup: Vec<(i32, i32, Vec<Option<f64>>)> = Vec::with_capacity(rows.len());
    for row in rows {
        if let Some(last) = dedup.last_mut() {
            if last.0 == row.0 && last.1 == row.1 {
                *last = row;
                continue;
            }
        }
        dedup.push(row);
    }
    let mut rows = dedup;

    // A quarter that came due unfiled enters at its deadline with nothing.
    if let Some(first_q) = rows.iter().map(|(_, q, _)| *q).min() {
        let mut earliest: HashMap<i32, i32> = HashMap::new();
        for (p, q, _) in &rows {
            earliest
                .entry(*q)
                .and_modify(|e| *e = (*e).min(*p))
                .or_insert(*p);
        }
        let horizon_q = quarter_index(kwargs.horizon_days);
        for q in first_q..=horizon_q {
            let due = deadline_days(q);
            if due <= kwargs.horizon_days && earliest.get(&q).is_none_or(|p| *p > due) {
                rows.push((due, q, vec![None; width]));
            }
        }
        rows.sort_by_key(|(p, q, _)| (*p, *q));
    }

    // One pass: per publish date, absorb that day's filings, then emit what
    // the market knew — the newest period and each offset's visible value.
    let vintages = {
        let mut count = 0usize;
        let mut last = i32::MIN;
        for (p, _, _) in &rows {
            if *p != last {
                count += 1;
                last = *p;
            }
        }
        count
    };
    let mut out_publish: Vec<i32> = Vec::with_capacity(vintages);
    let mut out_current: Vec<i32> = Vec::with_capacity(vintages);
    let mut out_values: Vec<Vec<Option<f64>>> = (0..offsets * width)
        .map(|_| Vec::with_capacity(vintages))
        .collect();

    let mut latest: HashMap<i32, Vec<Option<f64>>> = HashMap::new();
    let mut current = i32::MIN;
    let mut index = 0usize;
    while index < rows.len() {
        let day = rows[index].0;
        while index < rows.len() && rows[index].0 == day {
            let (_, q, values) = &rows[index];
            current = current.max(*q);
            latest.insert(*q, values.clone());
            index += 1;
        }
        out_publish.push(day);
        out_current.push(current);
        for k in 0..offsets {
            let seen = latest.get(&(current - k as i32));
            for j in 0..width {
                out_values[k * width + j].push(seen.and_then(|v| v[j]));
            }
        }
    }

    let mut fields: Vec<Series> = Vec::with_capacity(2 + offsets * width);
    fields.push(
        Int32Chunked::from_vec(kwargs.published.as_str().into(), out_publish)
            .into_series()
            .cast(&DataType::Date)?,
    );
    fields.push(Int32Chunked::from_vec("_current".into(), out_current).into_series());
    for k in 0..offsets {
        for (j, line) in kwargs.lines.iter().enumerate() {
            let name = if k == 0 {
                line.clone()
            } else {
                format!("{line}_{k}")
            };
            fields.push(
                Float64Chunked::from_iter_options(
                    name.as_str().into(),
                    std::mem::take(&mut out_values[k * width + j]).into_iter(),
                )
                .into_series(),
            );
        }
    }
    Ok(StructChunked::from_series("vintage".into(), vintages, fields.iter())?.into_series())
}
