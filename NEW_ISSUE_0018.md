source: pr-20268_a
# Centralize Spark `format_string` Format-Flag Compatibility Rules

## Summary

`datafusion/spark/src/function/string/format_string.rs` currently enforces some format-flag compatibility rules directly inside specific numeric formatting functions. The most immediate example is the rejection of the grouping separator flag `,` with scientific notation conversions, which is duplicated in both `format_float` and `format_decimal`.

This issue proposes moving unsupported flag/conversion combinations into a dedicated validation helper or validation layer so that Spark/Java compatibility rules live in one place.

## Problem

The formatter currently checks the `','` + scientific-notation incompatibility inline in two separate functions:

- `format_float`
- `format_decimal`

That duplication has a few costs:

- the rule is easier to miss when adding new formatter paths
- behavior can drift if one path is updated and the other is not
- compatibility rules are harder to discover because they are embedded in execution code rather than expressed declaratively

Today this is a small duplication. Over time it becomes a pattern unless there is a clear place for conversion/flag validation to live.

## Why This Matters

- Spark/Java compatibility rules should be explicit and centralized.
- Reviewers should be able to audit unsupported flag combinations without scanning multiple formatting implementations.
- Future formatter additions are less likely to miss validation if the checks are attached to `ConversionSpecifier` or a dedicated validator.

## Proposed Direction

Introduce a small validation helper that runs before formatting starts.

Possible shapes:

- a `ConversionSpecifier` method such as `validate_numeric_flag_compatibility()`
- a helper that returns a `Result<()>` for unsupported combinations
- a validation phase invoked once after parsing and before dispatching into type-specific formatting

The helper should own Spark-visible error construction for unsupported combinations so each formatter path does not need to rebuild the same message.

## Non-Goals

- Do not change current user-visible behavior or error text unless required for consistency.
- Do not collapse all formatting logic into a central dispatcher as part of this issue.
- Do not introduce speculative compatibility rules that are not already required by Spark/Java behavior.

## Scope

In scope:

- extracting duplicated flag/conversion validation
- using the shared validator from float and decimal formatting paths
- making the compatibility rule easier to extend for future formatter work

Out of scope:

- broader numeric formatting refactors unrelated to validation
- feature additions for currently unsupported formatter behavior

## Acceptance Criteria

- The `','` + scientific-notation incompatibility is defined in one place.
- `format_float` and `format_decimal` rely on the shared validation instead of duplicating the rule.
- Existing behavior and error messages remain covered by tests.
- The resulting structure makes it straightforward to add future compatibility checks without scattering them across formatting functions.

## Suggested Test Coverage

- float scientific conversions with `%,e` and `%,E`
- decimal scientific conversions with `%,e` and `%,E` if applicable
- precision variants such as `% ,.0e`-style grouped scientific inputs after normalizing to the exact supported syntax used in current tests
- a regression case proving the shared validation path is used consistently across float and decimal formatting

## Relevant Code

- `datafusion/spark/src/function/string/format_string.rs`
- `ConversionSpecifier`
- `format_float`
- `format_decimal`

## Implementation Notes

This is a small-to-medium refactor with a good payoff. It is tightly scoped, behavior-preserving, and directly addresses a concrete duplication introduced by the recent grouping-separator work.