source: refactor-byte-accounting-04-22688a
# Characterize StringView `append_with` failure-state semantics

## Problem

`StringViewArrayBuilder::try_append_with` and `StringViewArrayBuilder::append_with` contain near-duplicate finalization logic, but they differ around error handling and rollback. The infallible `append_with` path panics on overflow and may leave partially written state if that panic is caught. The fallible `try_append_with` path records writer errors and truncates `in_progress` before returning `Err`.

Before sharing finalization code, the intended state after a failed infallible append must be explicit and tested.

Affected file:
- `datafusion/functions/src/strings.rs`

## Why it matters

A naive refactor could make `append_with` delegate to `try_append_with(...).expect(...)`, which would silently change caught-panic behavior by rolling back `in_progress`. That is observable to callers that recover from panic and continue using the builder.

This is a semantic-risk refactor, not just cleanup. Tests should lock the contract before code is shared.

## Invariant / desired behavior

- `try_append_with` remains transactional for recoverable builder errors: on `Err`, no row is appended and partial row bytes written for that call are removed from `in_progress`.
- `append_with` preserves its existing infallible/panic semantics, including builder state after a caught panic.
- Any later shared helper must not blur the boundary between fallible rollback behavior and infallible panic behavior.

## Proposed direction

Add focused characterization tests around the two failure paths before extracting shared code:

1. A fallible `try_append_with` error path that proves partial spilled bytes are rolled back and the builder can still finish/append correctly.
2. An infallible `append_with` panic path using `catch_unwind` that documents the current post-panic builder state.

Keep tests private to the `strings.rs` test module. Do not add public testing hooks.

## Scope

### In

- Add unit tests for `StringViewArrayBuilder::try_append_with` rollback behavior.
- Add unit tests for `StringViewArrayBuilder::append_with` caught-panic behavior if a practical overflow trigger exists without huge allocation.
- Document in test names/comments that the purpose is preserving behavior before refactor.

### Out

- No production refactor yet.
- No public API changes.
- No change to error messages or panic messages.
- No attempt to force `append_with` to be transactional.
- No broad tests for every string function using the builder.

## Acceptance criteria

- [ ] Tests distinguish fallible rollback behavior from infallible panic behavior.
- [ ] Tests fail if `append_with` is changed to blindly delegate to `try_append_with(...).expect(...)` and that changes observable state.
- [ ] Tests do not require allocating multi-gigabyte strings or buffers.
- [ ] Existing successful append paths still pass.

## Tests / verification

- `cargo test -p datafusion-functions string_view_array_builder --lib`
- `cargo test -p datafusion-functions strings::tests::<new_test_name> --lib`
- If exact test filtering is awkward, run `cargo test -p datafusion-functions strings --lib`.

## Notes / open questions

- The main open question is how to trigger `append_with` overflow without impractical memory use. If no small trigger exists, document that and only test the fallible rollback path plus the success-path helper in the follow-up issue.
