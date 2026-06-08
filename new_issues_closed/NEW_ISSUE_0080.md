created #23665
source: pr-22714_a
# Document decimal AVG wrapping arithmetic behind explicit helpers

## Problem

Decimal `AVG` now widens its intermediate sum type through `avg_sum_data_type`, and the scalar, grouped, state-field, convert-to-state, and distinct paths mostly share that contract. However, the implementation still uses raw `add_wrapping` / `sub_wrapping` calls across the decimal AVG code paths.

Affected code includes:

- `datafusion/functions-aggregate/src/average.rs`
  - `decimal_sum_as`
  - scalar decimal accumulator `update_batch`, `merge_batch`, and `retract_batch`
  - grouped decimal accumulator `update_batch` and `merge_batch`
- `datafusion/functions-aggregate-common/src/aggregate/avg_distinct/decimal.rs`
  - distinct decimal AVG re-summing of stored distinct input values

The current behavior may be correct for the widened/headroom contract, but the raw wrapping calls make that contract hard to audit. A future change could reuse the same pattern in a context where wrapping is not safe.

## Why it matters

Decimal `AVG` has a correctness-sensitive invariant:

> The intermediate sum type must be wide enough for the documented AVG contract, or AVG must report an explicit overflow. Wrapping arithmetic must not silently corrupt results by accident.

Today this invariant is explained near `avg_sum_data_type`, but not at each arithmetic boundary. Readers must infer that every `add_wrapping` is intentional and protected by the widened state type. That makes review harder and increases regression risk when decimal state selection, partial aggregation, or distinct AVG changes.

## Invariant / desired behavior

- Decimal AVG update, retract, merge, and distinct re-sum paths use one clearly named accumulation contract.
- Wrapping arithmetic, if retained, is isolated behind helper functions whose names/comments state why it is safe for AVG's widened/headroom state.
- Partial-state merge follows the same contract as direct batch update.
- Existing SQL-visible return types and ordinary AVG behavior do not change.

## Proposed direction

Introduce small internal helpers near the decimal AVG implementation, for example:

```rust
fn add_avg_sum_widened<I, S>(sum: S::Native, value: I::Native) -> S::Native
where
    I: DecimalType + ArrowNumericType,
    S: DecimalType + ArrowNumericType,
    I::Native: Into<S::Native>,
```

and, if useful for merge/retract:

```rust
fn add_avg_state_sum<S>(sum: S::Native, value: S::Native) -> S::Native
fn sub_avg_sum_widened<I, S>(sum: S::Native, value: I::Native) -> S::Native
```

These helpers may still use `add_wrapping` / `sub_wrapping`, but their doc comments should tie the behavior to `avg_sum_data_type` and `AVG_SUM_HEADROOM_DIGITS`. If any path is not protected by that contract, use checked arithmetic and return a DataFusion overflow error there instead.

Keep the change narrow: this is primarily a readability and invariant-documentation refactor, not a redesign of decimal aggregate arithmetic.

## Scope

### In

- Replace direct decimal AVG `add_wrapping` / `sub_wrapping` calls with named helper calls.
- Document the widened/headroom contract at the helper boundary.
- Ensure scalar AVG, grouped AVG, partial-state merge, retract, and distinct AVG use the helper or an equivalent documented contract.
- Add or adjust focused tests only where current coverage does not exercise the helper paths.

### Out

- Do not change public SQL return types.
- Do not redesign all decimal aggregates or `SUM` semantics.
- Do not claim Decimal32/Decimal64/Decimal128 AVG currently accumulates in the narrow input type; current code already widens many of these paths.
- Do not add broad checked arithmetic if it causes avoidable overhead and the existing widened/headroom contract is sufficient.
- Do not change `DistinctSumAccumulator` behavior for `SUM(DISTINCT)` as part of this issue.

## Acceptance criteria

- [ ] Decimal AVG code paths no longer call raw `add_wrapping` / `sub_wrapping` directly, except inside a small documented helper.
- [ ] Helper names/comments state the AVG widened/headroom contract and reference `avg_sum_data_type` or equivalent state-type selection.
- [ ] Scalar, grouped, merge, retract, and distinct decimal AVG paths use the documented helper/contract.
- [ ] State field types, `state()`, `convert_to_state`, and `merge_batch` continue to agree on the sum type.
- [ ] Existing decimal AVG regression tests still pass.
- [ ] Any newly added overflow behavior is explicitly tested and documented.

## Tests / verification

Run focused tests first:

```bash
cargo test -p datafusion-functions-aggregate avg
cargo test -p datafusion-functions-aggregate-common avg_distinct
cargo test -p datafusion-sqllogictest --test sqllogictests decimal
```

Relevant existing coverage includes:

- `datafusion/functions-aggregate/src/average.rs` unit tests for widened scalar/grouped state types and convert-to-state roundtrip.
- `datafusion/functions-aggregate-common/src/aggregate/avg_distinct/decimal.rs` distinct AVG widening tests.
- `datafusion/sqllogictest/test_files/decimal.slt` decimal AVG overflow-regression cases.

## Notes / open questions

- Decimal256 has no wider exact native state. If a future patch changes its overflow behavior from documented wrapping/headroom to checked overflow, that should be treated as a behavior change with dedicated tests.
