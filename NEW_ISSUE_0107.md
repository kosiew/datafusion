source: refactor-byte-accounting-04-22688a
# Extract shared StringView append-with success finalization

## Problem

`StringViewArrayBuilder::try_append_with` and `StringViewArrayBuilder::append_with` duplicate the success-path logic that turns a completed `StringViewWriter` into a `StringViewArrayBuilder` row.

Both paths:
- create a `StringViewWriter`,
- run the caller closure,
- destructure writer state to release the borrow,
- append either an inline view or a long view.

The duplication makes future maintenance risky because fixes to inline/long view finalization can land in one path but not the other. The paths differ only where failure semantics matter.

Affected file:
- `datafusion/functions/src/strings.rs`

## Why it matters

StringView row finalization encodes Arrow view fields (`length`, `prefix`, `buffer_index`, `offset`). Small inconsistencies can produce invalid arrays, incorrect IPC behavior, or different overflow handling between fallible and infallible APIs.

Sharing only the success finalization keeps the code easier to audit while preserving the critical semantic difference between `try_append_with` rollback and `append_with` panic behavior.

## Invariant / desired behavior

- Inline rows are finalized with `make_view(&inline_buf[..inline_len], 0, 0)`.
- Long rows are finalized from bytes already written to `in_progress[start..]`.
- Fallible finalization checks row length, offset, and buffer count through the existing `try_long_view_parts` path and rolls back on `Err`.
- Infallible finalization preserves current panic behavior and must not gain fallible rollback as a side effect.
- No public interfaces change.

## Proposed direction

Extract a small private helper for the common success shape, without changing error policy:

- Keep writer construction and error handling in `try_append_with` / `append_with`.
- Consider a helper around the shared inline/long match only.
- If a helper needs to differ on checked vs panicking conversion, use two tiny private helpers with parallel names rather than a generic abstraction that hides error policy.
- Avoid making `append_with` call `try_append_with`.

A safe shape is likely:

- one private fallible finalizer used only by `try_append_with`, returning `Result<()>` and allowing caller rollback on error; and/or
- one private infallible finalizer used only by `append_with`, panicking exactly as today.

Only share code where the same state transition and failure policy are truly identical.

## Scope

### In

- Extract private helper(s) inside `impl StringViewArrayBuilder`.
- Preserve existing `try_append_with` rollback behavior.
- Preserve existing `append_with` panic behavior.
- Keep comments around borrow release if destructuring remains necessary.
- Keep current public methods and trait implementation unchanged.

### Out

- No change to `BulkNullStringArrayBuilder`.
- No behavior change for `append_value`, `append_byte_map`, or `finish`.
- No broad StringView builder redesign.
- No new abstraction shared with `GenericStringArrayBuilder`.
- No performance-motivated rewrite beyond removing local duplication.

## Acceptance criteria

- [ ] `try_append_with` and `append_with` no longer duplicate the full inline/long finalization match.
- [ ] The extracted helper(s) make error policy explicit in names, return type, or call site.
- [ ] `append_with` does not delegate to `try_append_with(...).expect(...)`.
- [ ] Characterization tests from `NEW_ISSUE_01.md` still pass.
- [ ] Existing StringView builder success tests still pass.
- [ ] No public API or trait signature changes.

## Tests / verification

- `cargo test -p datafusion-functions strings --lib`
- `cargo test -p datafusion-functions string_view_builder --lib`
- `cargo test -p datafusion-functions --lib` if the refactor touches shared builder behavior used by multiple string functions.

## Notes / open questions

- Prefer a small duplication if sharing would obscure the difference between recoverable `Err` and panic semantics.
- If `NEW_ISSUE_01.md` cannot produce a practical caught-panic characterization test, keep this refactor more conservative: extract only code that is impossible to affect rollback/panic state.
