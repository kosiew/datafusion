source: refactor-byte-accounting-04-22688a
# Evaluate shared lpad/rpad padding emitters without changing semantics

## Problem

`lpad` and `rpad` now contain near-mirrored implementations for scalar and row-wise padding. The duplication spans:

- scalar ASCII fast paths,
- scalar Unicode fast paths,
- row-wise optional fill handling,
- null propagation,
- target-length validation,
- empty-fill behavior,
- Unicode boundary/truncation handling,
- repeated fill emission.

Affected files:

- `datafusion/functions/src/unicode/lpad.rs`
- `datafusion/functions/src/unicode/rpad.rs`
- possibly `datafusion/functions/src/unicode/common.rs` for private shared helpers

The current code is readable locally, but fixes to one side can drift from the other. A broad rewrite would be risky because the two functions differ exactly where users notice behavior: which side padding is written on and which side is preserved when truncating.

## Why it matters

`lpad` / `rpad` are SQL-visible Unicode string functions. Small refactor mistakes can change query results for edge cases:

- multibyte input strings,
- multibyte fill strings,
- empty fill strings,
- negative target lengths,
- target lengths larger than `i32::MAX`,
- null combinations across string, length, and fill inputs,
- scalar-vs-array argument paths,
- `Utf8`, `LargeUtf8`, and `Utf8View` output behavior.

The goal is maintainability, not new behavior. Any shared code must make direction-specific behavior explicit rather than hiding it behind a vague abstraction.

## Invariant / desired behavior

- `lpad` and `rpad` outputs remain byte-for-byte identical to current behavior for all supported string types.
- Null policy remains unchanged: any null input row produces a null output row.
- Negative target lengths continue to produce an empty string for non-null rows.
- Target lengths greater than `i32::MAX` continue to return the same function-specific error text.
- Empty fill strings continue to return the original string when padding is needed, subject to current truncation rules.
- Unicode truncation continues to respect character boundaries.
- No public APIs, signatures, SQL behavior, or error messages change.

## Proposed direction

Start with characterization before sharing code.

1. Add or confirm focused tests that cover the shared-risk matrix for both `lpad` and `rpad`.
2. Extract only small private helpers where behavior is truly identical, such as:
   - target-length normalization,
   - repeated fill planning/emission,
   - common Unicode fill-buffer construction.
3. Keep direction-specific behavior explicit at call sites or in a tiny enum with obvious names, e.g. `PadSide::Left` / `PadSide::Right`.
4. Avoid one large generic padding engine unless tests prove it preserves every edge case and the result is easier to audit than the current files.

Preferred first patch: centralize target-length normalization and remove obvious duplicated inner-loop details. Defer full emitter unification unless duplication continues to cause bugs.

## Scope

### In

- Private helper extraction only.
- Tests for `lpad` and `rpad` edge-case parity before and after refactor.
- Preserve current scalar fast-path dispatch.
- Preserve current output type behavior for `Utf8`, `LargeUtf8`, and `Utf8View`.
- Keep error messages exactly the same.

### Out

- No public API changes.
- No SQL behavior changes.
- No new padding features.
- No change to function signatures or coercion rules.
- No broad rewrite of string builders.
- No attempt to merge unrelated string functions.
- No performance-only rewrite unless benchmarks are added separately.

## Acceptance criteria

- [ ] A reviewer can identify the direction-specific behavior for left vs right padding without reading a large generic engine.
- [ ] Shared helper(s) remove duplicated logic only where branch behavior and error policy are identical.
- [ ] `lpad` and `rpad` retain exact outputs for ASCII, Unicode, empty fill, negative length, zero length, null, and truncation cases.
- [ ] Error text for oversized target length remains function-specific and unchanged.
- [ ] No public interface changes.
- [ ] The final diff is local to `lpad.rs`, `rpad.rs`, and optional private helpers/tests.

## Tests / verification

Run targeted Rust tests in `datafusion-functions`:

- `cargo test -p datafusion-functions unicode::lpad --lib`
- `cargo test -p datafusion-functions unicode::rpad --lib`
- If filters differ, run `cargo test -p datafusion-functions lpad --lib` and `cargo test -p datafusion-functions rpad --lib`.

Test cases should include:

- ASCII string + ASCII fill for both left and right padding.
- Unicode input where truncation lands on a character boundary.
- Unicode fill repeated cyclically.
- Empty fill string.
- Negative, zero, and positive target lengths.
- Target length greater than `i32::MAX` and exact error text.
- Null string, null length, and null fill.
- `Utf8`, `LargeUtf8`, and `Utf8View` where applicable.

If the refactor changes SQL-visible behavior accidentally, add/adjust SQLLogicTest coverage under `datafusion/sqllogictest/test_files/` before proceeding.

## Notes / open questions

- Full `lpad`/`rpad` unification may not be worth it if a smaller helper removes the risky duplication.
- Prefer explicit duplicated code over a clever abstraction that makes Unicode truncation or padding side hard to verify.
