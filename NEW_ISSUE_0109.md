source: refactor-byte-accounting-04-22688a
# Design cautious null-aware bulk-builder loop helpers

## Problem

Several string and Unicode functions repeat the same bulk-null loop pattern:

1. compute a union null buffer from inputs,
2. append a placeholder row when the output row is null,
3. use unchecked array access after the null check,
4. append a computed string row through `BulkNullStringArrayBuilder`,
5. finish with the precomputed null buffer.

Affected files include:

- `datafusion/functions/src/string/repeat.rs`
- `datafusion/functions/src/string/split_part.rs`
- `datafusion/functions/src/core/overlay.rs`
- `datafusion/functions/src/unicode/translate.rs`
- `datafusion/functions/src/unicode/reverse.rs`
- `datafusion/functions/src/unicode/substr.rs`
- `datafusion/functions/src/unicode/lpad.rs`
- `datafusion/functions/src/unicode/rpad.rs`
- possible private helper location: `datafusion/functions/src/strings.rs` or function-local modules

The duplication is real, but a generic helper can make code less safe if it hides each function's null policy, error policy, or `unsafe` preconditions.

## Why it matters

These loops sit on a correctness boundary:

- Null propagation must match SQL function semantics.
- `unsafe value_unchecked` calls are only valid after the right null and bounds checks.
- Builder placeholder count must stay aligned with the null buffer passed to `finish`.
- Function-specific errors must still take precedence where they currently do.

A broad helper or macro could reduce duplicated code but accidentally obscure when unchecked access is valid. The refactor should only happen if it improves auditability.

## Invariant / desired behavior

- For every row, output nullness remains exactly the current union of relevant input nulls.
- Placeholder rows are appended exactly for rows masked as null in the final null buffer.
- Non-null rows use unchecked access only after a proven matching null check and valid row index.
- Function-specific validation and error order stay unchanged.
- Builder overflow errors continue to propagate through `Result`.
- No public APIs or SQL behavior change.

## Proposed direction

Do this as a design-and-prototype refactor, not a sweeping mechanical rewrite.

1. Pick one or two low-risk call sites with identical null policy, such as unary `reverse` and binary `repeat`, to test helper shape.
2. Prefer a small private helper only if it makes the unsafe invariant clearer at the call site.
3. Keep arity-specific helpers explicit. For example:
   - one helper for unary string arrays,
   - one helper for binary string/int arrays,
   - avoid variadic/generic abstractions that erase input types.
4. Require call sites to provide the row computation closure and keep function-specific validation visible.
5. Do not force all string functions through the helper. Leave bespoke loops where null/error semantics differ or readability worsens.

A good helper should read like: "for each non-null row proven by this null buffer, call this closure with safe values or clearly documented unchecked values." If it cannot state that invariant simply, keep the loop local.

## Scope

### In

- Private helper or small macro exploration for repeated null-aware builder loops.
- Apply only to call sites where null policy is exactly union-of-input-nulls and error order remains unchanged.
- Document the unsafe invariant at the helper boundary.
- Add/keep targeted tests for functions migrated to the helper.
- Preserve current builder APIs and function signatures.

### Out

- No public API changes.
- No broad rewrite of all string functions.
- No abstraction that supports every arity/type combination at the cost of clarity.
- No change to null semantics or error precedence.
- No change to `BulkNullStringArrayBuilder` trait unless separately justified.
- No performance claims without benchmarks.

## Acceptance criteria

- [ ] Any introduced helper has a narrow private API and an explicit null/unsafe invariant in its docs or comments.
- [ ] At least one migrated call site is shorter or clearer without hiding function-specific behavior.
- [ ] Call sites with distinct validation/error policy are intentionally left unchanged.
- [ ] Placeholder rows and final null buffer lengths remain aligned.
- [ ] Existing overflow propagation tests still pass.
- [ ] No public interface changes.

## Tests / verification

For each function migrated to a helper, run its targeted tests. Expected starting set:

- `cargo test -p datafusion-functions reverse --lib`
- `cargo test -p datafusion-functions repeat --lib`
- `cargo test -p datafusion-functions split_part --lib`
- `cargo test -p datafusion-functions overlay --lib`
- `cargo test -p datafusion-functions translate --lib`
- `cargo test -p datafusion-functions lpad --lib`
- `cargo test -p datafusion-functions rpad --lib`

Also run broader crate tests if the helper lands in shared code:

- `cargo test -p datafusion-functions --lib`

Test cases for migrated functions should include:

- all-null and mixed-null inputs,
- no-null inputs to exercise the fast unchecked loop,
- builder overflow propagation where existing failing-builder tests cover it,
- function-specific invalid inputs, preserving current error order,
- `Utf8`, `LargeUtf8`, and `Utf8View` variants where supported.

## Notes / open questions

- A macro may preserve type-specific readability better than a highly generic function, but macros can hide control flow. Choose only if the call sites remain obvious.
- `substr` and `split_part` have zero-copy/view-specific paths; those may not fit a common helper cleanly.
- If the first two migrated call sites are not clearly better, stop and keep the local loops.
