created #24058
source: pr-23718_a
# Refactor numeric sign and padding in Spark format_string

## Problem
`datafusion/spark/src/function/string/format_string.rs` has separate sign and width-padding logic in `format_float`, `format_decimal`, and integer decimal formatting. These paths all need to preserve the same Java/Spark formatting contract for flags such as `(`, `+`, space, `0`, `-`, and width, but they duplicate the final assembly of `prefix + number + suffix`.

A recent decimal bug was caused by this drift: `format_decimal` missed `negative_in_parentheses` suffix handling that already existed in `format_float`. The bug is now fixed, but the underlying refactor opportunity remains: the shared final-assembly invariant is still encoded separately in multiple paths.

## Why it matters
Duplicated tail-formatting logic makes future fixes easy to apply to one numeric type but miss another. This is a correctness risk for SQL-visible `format_string` behavior across Float16/32/64, Decimal128/256, and integer inputs.

## Invariant / desired behavior
After a numeric value has been converted into its unsigned/absolute textual `number`, all numeric formatting paths should apply sign, parentheses suffix, zero padding, left adjustment, and width using one canonical helper.

Type-specific number generation must remain separate:
- float finite / NaN / Infinity behavior
- decimal exact BigDecimal formatting
- integer base formatting
- `%g` / scientific precision decisions

## Proposed direction
Add a narrow helper for final numeric assembly and padding, not a broad helper that tries to merge float and decimal formatting.

Suggested helper responsibility:
- accept `prefix`, `number`, `suffix`
- account for `width`, `left_adj`, and `zero_pad`
- write the final string
- allow callers to disable zero padding when required, e.g. non-finite floats

Keep `format_float` and `format_decimal` responsible for producing `number`. Use the helper only after type-specific formatting is complete.

## Scope
### In
- Add one small helper in `ConversionSpecifier` for `prefix + number + suffix` width/padding assembly.
- Use it from `format_float` and `format_decimal`.
- Consider using it from signed integer decimal formatting if it reduces duplication without changing behavior.
- Preserve all existing outputs.
- Keep or add focused regression coverage where helper behavior could drift, especially negative parentheses with width and zero padding.

### Out
- Do not merge decimal and float number-generation logic.
- Do not change Spark/Java compatibility semantics.
- Do not rewrite parsing or validation of format specifiers.
- Do not alter scientific / compact `%g` precision behavior.
- Do not introduce a broad abstraction with many primitive flags beyond final assembly/padding.

## Acceptance criteria
- [ ] `format_float` and `format_decimal` use the same final sign/suffix/width-padding helper.
- [ ] Existing float, decimal, and integer `format_string` tests pass unchanged unless expected strings are proven wrong against Spark/Java.
- [ ] Negative-parentheses decimal formatting remains covered for no-width and width cases, confirming the existing bug fix stays protected.
- [ ] Helper has a narrow contract documented by its name/signature or a short comment.
- [ ] Refactor is behavior-preserving except for any separately justified bug fixes.

## Tests / verification
- Run targeted Rust tests for Spark format string:
  - `cargo test -p datafusion-spark format_string --lib`
- At minimum, include/keep regression coverage for:
  - Decimal128 negative `%(,.2f`
  - Decimal128 negative `%(,15.2f`
  - Float negative `%(,15.2f`
  - zero padding with sign/parentheses where supported
- If integer formatting is touched, run existing integer `format_string` tests and add a focused case only if current coverage does not protect sign + width placement.

## Notes / open questions
- Decimal256 uses the same `format_decimal` path as Decimal128. A small Decimal256 regression case may be useful if the refactor touches decimal dispatch or tests are otherwise too Decimal128-specific.
