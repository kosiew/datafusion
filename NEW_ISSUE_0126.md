source: pr-24115_a
# Unify integer decimal sign and width assembly with numeric formatting

## Problem

`datafusion/spark/src/function/string/format_string.rs` has one canonical final-assembly helper, `ConversionSpecifier::write_numeric_parts`, for float and decimal formatting. It assembles a sign/parenthesis prefix, numeric body, optional suffix, and width padding.

Integer `%d` formatting still bypasses that invariant. `format_signed` computes a sign prefix/suffix, reduces the width, delegates to `format_unsigned`, then splits and reassembles leading spaces. `format_unsigned` independently implements left adjustment, zero padding, and space padding.

This leaves three implementations of the same final-assembly rules. A prior decimal formatting bug omitted the negative-parentheses suffix because equivalent final assembly had drifted between numeric paths. The signed integer path can drift independently for `(`, `+`, space, `0`, `-`, and width combinations; the unsigned integer path independently implements its own zero, left, and space padding.

## Why it matters

`format_string` is SQL-visible Spark compatibility behavior. Divergent implementations make a flag combination correct for floats/decimals but incorrect for `%d`, and force future fixes to be replicated across multiple paths. The current integer path is also harder to audit because width is adjusted before formatting and the result is parsed again to place the sign after leading spaces.

## Invariant / desired behavior

For every numeric conversion, final rendering has one meaning:

- The rendered value is `prefix + number + suffix`.
- Width is measured over all three parts.
- `-` pads spaces on the right.
- `0` pads between `prefix` and `number`; a suffix, including `)`, remains last.
- Otherwise, width pads spaces on the left.
- Sign selection remains type-specific, but its placement and width behavior are shared.

For signed integer `%d`, the result must remain compatible with the Java/Spark contract for grouping, `+`, space, `(`, `0`, `-`, and width. `UInt*` inputs are a DataFusion extension, not a Java/Spark type: preserve their current `%d` behavior for grouping, `0`, `-`, and width. Do not add sign-policy behavior for unsigned values as part of this refactor.

## Proposed direction

Refactor `%d` formatting to produce numeric parts before padding:

1. Keep integer value conversion and decimal grouping local to the integer formatter.
2. Derive the signed integer prefix/suffix using the same sign rules as float and decimal paths.
3. Pass the integer numeric body and those parts to `write_numeric_parts` with zero padding allowed.
4. Retain separate hexadecimal and octal alternate-prefix behavior unless their semantics can be expressed by the same helper without broadening this issue.
5. Remove the width subtraction, post-format leading-space scan, and duplicate `%d` padding logic once the shared helper owns final assembly.

The helper should remain private and narrowly own final assembly only; it should not absorb integer conversion, grouping, or sign-selection policy.

## Scope

### In
- Refactor signed and unsigned `%d` final assembly in `datafusion/spark/src/function/string/format_string.rs` to use the numeric-parts invariant.
- Preserve current decimal grouping behavior for integer `%d`.
- Add regression coverage for the shared final-assembly contract across float, decimal, signed integer, and unsigned integer inputs, preserving the existing unsigned-input policy.
- Cover both `Decimal128` and `Decimal256` where decimal regression coverage is extended.

### Out
- Changing supported conversion specifiers or parser validation.
- Changing Java/Spark compatibility semantics.
- Refactoring `%x` / `%X` / `%o` assembly beyond a small prerequisite needed to keep `%d` code cohesive.
- Changing non-finite float zero-padding policy.
- General formatting performance work unrelated to removing duplicate assembly.

## Acceptance criteria

- [ ] Signed `%d` no longer subtracts sign/suffix width and reparses formatted output to restore sign placement.
- [ ] Signed and unsigned `%d` use the same final prefix/number/suffix width-assembly owner as floats and decimals, or an equally single canonical owner with documented equivalent semantics.
- [ ] `%d` preserves correct output for negative values with `(`, `+`, space, `0`, `-`, and literal width, including combinations of parentheses, zero padding, and width.
- [ ] Signed `%d` preserves correct output for positive values with `+`, space, `0`, `-`, grouping, and width.
- [ ] Unsigned `%d` preserves existing output for grouping, `0`, `-`, and width; this refactor does not introduce `+`, space, or parentheses sign policy for `UInt*`.
- [ ] Existing float behavior keeps finite-only zero padding; `NaN` and infinities remain space-padded when width is requested.
- [ ] Existing decimal behavior preserves a closing `)` after zero padding for negative parenthesized values.
- [ ] No duplicate final width-padding branches remain for decimal `%d` formatting.

## Tests / verification

- Add focused Rust unit tests near the existing `format_string` tests for direct formatter behavior across `Int64`, `UInt64`, `Decimal128`, `Decimal256`, and `Float64`.
- Add SQLLogicTest coverage in `datafusion/sqllogictest/test_files/spark/string/format_string.slt` for SQL-visible representative cases:
  - negative `%d` with `(`, `0`, and width;
  - positive/negative signed `%d` with `+` or space and width;
  - unsigned `%d` with grouping, zero padding, left adjustment, and width;
  - negative decimal `%(...f` with `0` and width, asserting the trailing `)`;
  - `NaN` and positive/negative infinity float formatting with `0` and width, asserting space rather than zero padding.
- Run `cargo test -p datafusion-spark --all-features`.
- Run the focused Spark format-string SQLLogicTest through the repository SQLLogicTest harness.

## Notes / open questions

- Confirm Java `Formatter` output for selected signed combined-flag cases before recording expected values; preserve that compatibility contract rather than infer it from the current implementation.
- Java/Spark cannot establish `UInt*` behavior. Record the current behavior in focused tests before the refactor, then preserve it.
- `format_unsigned` also serves hexadecimal and octal conversions. Keep their alternate-prefix semantics stable when extracting `%d` final assembly.
