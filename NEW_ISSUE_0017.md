source: pr-20268_a
# Refactor Spark `format_string` Numeric Formatting Paths

## Summary

`datafusion/spark/src/function/string/format_string.rs` currently formats integers, floats, and decimals through partially separate code paths. Recent grouping-separator work extended `format_float` and `format_decimal`, while decimal integer formatting in `format_unsigned` still follows its own implementation. The result is a formatter surface with overlapping logic for grouping, sign handling, width/padding, and some flag behavior, but without a shared abstraction that makes those rules explicit.

This issue proposes a scoped refactor to reduce duplication across numeric formatters without forcing all numeric types through a single overly-generic implementation.

## Problem

The current implementation has three distinct numeric formatting paths:

- `format_unsigned` for integer-like formatting
- `format_float` for `f64`
- `format_decimal` for `Decimal128`

These paths share some responsibilities:

- choosing or applying a sign representation
- applying grouping separators
- applying width, left-adjustment, and zero-padding
- enforcing some conversion/flag compatibility rules

At the same time, they also differ materially:

- floats handle finite vs non-finite values
- floats and decimals support fixed/scientific/compact formatting
- decimals rely on `BigDecimal`-specific formatting and rounding behavior
- signed integers delegate through `format_unsigned`

That means a full unification would likely add too much abstraction pressure, but the current split also makes it easy for behavior to drift when adding or adjusting Spark-compatible formatting rules.

## Why This Matters

- Similar formatting concerns are implemented multiple times, increasing maintenance cost.
- Spark-compatibility fixes are harder to reason about when sign, grouping, and padding behavior are spread across separate paths.
- Future formatter work will be easier to review if common behavior is expressed through a smaller set of shared helpers or phases.

## Proposed Direction

Refactor toward a shared numeric-formatting pipeline or a smaller set of shared helpers, while keeping type-specific formatting decisions local.

One plausible direction:

1. Keep type-specific number generation local.
   Float and decimal code should still decide how to produce the core numeric representation for fixed/scientific/compact output.

2. Extract shared post-processing helpers.
   Examples:
   - grouping insertion for decimal-style output
   - sign decoration
   - width and zero-padding application
   - final assembly of prefix/number/suffix

3. Reuse those helpers across integer, float, and decimal formatting.
   This reduces duplication without pretending the source formatting logic is identical.

## Non-Goals

- Do not change Spark-visible formatting semantics as part of the refactor.
- Do not rewrite all numeric formatting into a single monolithic helper if that obscures type-specific behavior.
- Do not expand formatter feature support in the same change unless needed to preserve existing semantics.

## Scope

In scope:

- identifying duplicated numeric post-processing behavior
- extracting shared helpers where behavior is already intended to match
- simplifying call sites in `format_unsigned`, `format_float`, and `format_decimal`
- preserving existing behavior with regression coverage

Out of scope:

- introducing new format conversions
- changing Spark compatibility rules
- broad cleanup of unrelated formatter paths such as strings, booleans, dates, or times

## Acceptance Criteria

- Shared numeric formatting concerns are expressed in reusable helpers or phases instead of repeated inline logic.
- `format_unsigned`, `format_float`, and `format_decimal` are easier to compare and review after the refactor.
- Existing grouping, sign, width, and padding behavior is preserved.
- Test coverage exists for representative integer, float, and decimal cases so refactoring does not change behavior accidentally.

## Suggested Test Coverage

- grouped integer formatting
- grouped fixed float formatting
- grouped fixed decimal formatting
- width + zero-padding + grouping interactions
- sign handling with grouping for positive and negative values
- compact/scientific cases that must not accidentally use grouping where unsupported

## Relevant Code

- `datafusion/spark/src/function/string/format_string.rs`
- `format_unsigned`
- `format_float`
- `format_decimal`
- `insert_thousands_separator`

## Implementation Notes

This is a medium-to-large cleanup task, not an urgent correctness fix. The main goal is to reduce maintenance overhead and make future Spark-formatting changes easier to implement safely. A phased refactor with small behavior-preserving extractions is preferable to a one-shot rewrite.