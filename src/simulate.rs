//! The account walk as a polars expression, under the framework's own policy.

use johnnybt_engine::bookkeeping::Plain;
use johnnybt_engine::plugin::{output_field, simulate as walk, SimulateKwargs};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn simulate_output(_input_fields: &[Field], kwargs: SimulateKwargs) -> PolarsResult<Field> {
    Ok(output_field("simulate", kwargs.positions, kwargs.fills))
}

/// Walk one account: one row per bar in, one struct per bar out.
///
/// Called under `group_by(account).agg(...)`: each group is one account and
/// polars runs them on its own thread pool. The kwargs say which input is
/// which and carry the market's rules and the account's terms; see
/// `johnnybt_engine::plugin::SimulateKwargs`.
#[polars_expr(output_type_func_with_kwargs=simulate_output)]
fn simulate(inputs: &[Series], kwargs: SimulateKwargs) -> PolarsResult<Series> {
    walk::<Plain>(inputs, &kwargs)
}
