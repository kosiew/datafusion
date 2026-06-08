created #23667
source: pr-23267_a
# Centralize higher-order list lambda evaluation helpers

## Problem
`array_filter`, `array_any_match`, and `array_first` each implement their own version of the same higher-order list execution pattern:

1. normalize a list-like argument to `List` / `LargeList`,
2. extract the flattened child values with slice-aware semantics,
3. evaluate a lambda once over the flattened values,
4. spread captured outer columns to flattened row cardinality with `list_values_row_number`,
5. map the flattened lambda result back to one output value per input row using adjusted offsets and null-row handling.

Some common pieces already live in `datafusion/functions-nested/src/lambda_utils.rs`, but the predicate-evaluation and row-remapping contracts are still duplicated in individual functions such as:

- `datafusion/functions-nested/src/array_filter.rs`
- `datafusion/functions-nested/src/array_any_match.rs`
- `datafusion/functions-nested/src/array_first.rs`

This makes subtle behavior easy to drift across functions: sliced lists, null list rows, empty lists, nullable elements, `List` vs `LargeList`, and coerced `FixedSizeList` / list-view inputs all need consistent treatment.

## Why it matters
Higher-order list functions are boundary code between logical lambda expressions and Arrow array layout. Small mistakes here usually surface as correctness bugs only for edge cases: sliced arrays, null rows with non-empty backing child values, empty rows, or captured outer columns.

Duplicated flatten/evaluate/spread logic also raises the cost of adding new higher-order array functions. Each new function must rediscover the same offset and null invariants, and reviewers must re-check the same contracts again.

A shared helper would improve:

- correctness consistency across higher-order list functions,
- reviewability for new functions,
- coverage of list-like input coercions,
- future support for additional Arrow list layouts.

## Invariant / desired behavior
All single-list higher-order functions that evaluate a lambda over list elements should share one canonical execution contract:

- lambda input cardinality equals the number of reachable flattened elements for the input rows,
- captured outer arrays are expanded to the same flattened cardinality using the list row mapping,
- sliced list arrays use offsets relative to the sliced child values,
- null list rows do not expose unreachable child values to the lambda after `clear_null_values` handling,
- empty and null rows map back deterministically according to each function's result semantics,
- `List` and `LargeList` behavior stays equivalent except for offset width,
- coercions from `FixedSizeList`, `ListView`, and `LargeListView` remain explicit and tested where advertised.

## Proposed direction
Introduce a small shared helper in `datafusion/functions-nested/src/lambda_utils.rs` for the common single-list lambda evaluation path.

The helper should not decide each function's final result semantics. Instead, it should centralize only the mechanical boundary work:

- validate/extract the `(list, lambda)` pair,
- materialize the list argument for `number_rows`,
- handle fully-null fast paths where appropriate or expose enough state for callers to do so,
- extract slice-aware flattened list values,
- evaluate the lambda over those values,
- spread captured arrays with `list_values_row_number`,
- validate the expected lambda result type when the caller asks for a boolean predicate,
- expose adjusted per-row ranges over the flattened result.

Possible shape:

- a `SingleListLambdaInput` / `EvaluatedListLambda` struct containing:
  - original list array,
  - flattened values array,
  - evaluated lambda result array,
  - row ranges based on adjusted offsets,
  - original null bitmap / row validity metadata.
- a boolean-predicate convenience wrapper for functions like `array_filter`, `array_any_match`, and `array_first`.

Then migrate one function at a time, preserving behavior.

## Scope
### In
- Add shared helper(s) for single-list higher-order lambda evaluation in `lambda_utils.rs` or a nearby module.
- Migrate `array_first`, `array_any_match`, and `array_filter` to use the helper where it reduces duplicated flatten/evaluate/spread code.
- Keep each function's result-specific logic local:
  - `array_filter`: build filtered list output,
  - `array_any_match`: reduce predicate ranges to boolean/null,
  - `array_first`: choose first matching element index and `take` it.
- Preserve existing public SQL behavior and aliases.
- Add focused regression tests for shared edge cases:
  - sliced `List`,
  - null list rows with non-empty child values,
  - empty list rows,
  - nullable elements and nullable predicates,
  - captured outer columns,
  - `List` and `LargeList` parity.
- Add coverage for any list-like coercions the helper claims to support.

### Out
- Do not change lambda evaluation from eager/vectorized to row-local short-circuiting.
- Do not add new SQL functions.
- Do not redesign the higher-order UDF trait API unless the helper cannot be implemented locally.
- Do not change documented null semantics for existing functions.
- Do not attempt broad support for unsupported Arrow layouts unless covered by explicit tests and contracts.

## Acceptance criteria
- [ ] `array_filter`, `array_any_match`, and `array_first` no longer duplicate the core flatten/evaluate/spread/captured-column logic.
- [ ] A single helper owns the contract for flattened lambda input cardinality and captured-column expansion.
- [ ] Existing tests for the migrated functions pass without behavior changes.
- [ ] New regression tests cover sliced lists, null rows, empty rows, nullable elements, captured outer columns, and `List` / `LargeList` parity through the shared path.
- [ ] Any advertised `FixedSizeList`, `ListView`, or `LargeListView` coercion behavior is either tested or explicitly not claimed by the helper.
- [ ] Error messages for non-boolean predicates in predicate-based functions remain clear and function-specific.

## Tests / verification
- Unit tests in `datafusion-functions-nested` for the helper and migrated functions.
- SQLLogicTests for representative SQL behavior if migration changes user-visible paths.
- Existing targeted checks:
  - `cargo test -p datafusion-functions-nested array_filter`
  - `cargo test -p datafusion-functions-nested array_any_match`
  - `cargo test -p datafusion-functions-nested array_first`
  - relevant `datafusion-sqllogictest` array test files.
- Run `cargo fmt --all` and targeted clippy if code changes are non-trivial.

## Notes / open questions
- `lambda_utils.rs` already contains shared pieces (`coerce_single_list_arg`, `single_list_lambda_parameters`, `extract_list_values`). The issue is not to replace those, but to extend sharing to the higher-risk lambda evaluation and row-range mapping path.
- The helper should make eager predicate evaluation explicit. Short-circuit evaluation, if desired later, should be a separate design issue because it changes execution semantics and optimization assumptions.
