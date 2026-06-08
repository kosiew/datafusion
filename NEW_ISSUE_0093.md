source: refactor-byte-accounting-02-22688a
# Refactor Issue 02: Add a Checked Writer Abstraction for String Builder `append_with`

## Summary
Introduce a fallible / checked writer path for `GenericStringArrayBuilder` `append_with`-style APIs so generated string output can enforce offset limits while bytes are written, instead of writing first and rolling back after detecting overflow.

## Context
`GenericStringArrayBuilder::try_append_with` lets UDFs emit a row through a `StringWriter` without first allocating a temporary `String`:

```rust
builder.try_append_with(|w| replace_into_writer(w, string, from, to))
```

Today the writer API is infallible:

```rust
pub(crate) trait StringWriter {
    fn write_str(&mut self, s: &str);
    fn write_char(&mut self, c: char);
}
```

Because `StringWriter` cannot fail, `try_append_with` can only check the final byte length after the closure has written into the builder buffer. On overflow it rolls back the buffer length and returns `DataFusionError`.

This works, but the overflow invariant is enforced late and requires speculative mutation plus rollback.

## Problem
The new fallible API masks an underlying abstraction mismatch:

- API name says `try_append_with`, but the closure receives an infallible writer.
- The writer cannot report overflow at the write that crosses the offset boundary.
- Large generated rows may write many bytes before the final offset check fails.
- Rollback logic becomes part of the builder correctness contract.
- Future UDF migrations using `try_append_with` inherit this speculative-write behavior.

The core invariant should ideally be enforced at the write boundary: every write that would make the current row exceed the offset type limit should fail immediately.

## Goal
Add a checked writer abstraction for generated string output so fallible append paths can validate cumulative byte length before extending the value buffer.

Possible API shape:

```rust
pub(crate) trait TryStringWriter {
    fn try_write_str(&mut self, s: &str) -> Result<()>;
    fn try_write_char(&mut self, c: char) -> Result<()>;
}
```

Then add a checked append method:

```rust
pub fn try_append_with_checked<F>(&mut self, f: F) -> Result<()>
where
    F: FnOnce(&mut GenericTryStringWriter<'_, O>) -> Result<()>;
```

Or replace current `try_append_with` if churn is acceptable inside the crate.

## Non-Goals
- Do not change SQL-visible string function behavior.
- Do not make `StringViewArrayBuilder` part of this issue unless a clear shared abstraction emerges.
- Do not migrate every UDF in the first patch.
- Do not expose a public API outside `datafusion/functions` unless required.
- Do not optimize allocation strategy beyond enforcing offset limits earlier.

## Proposed Scope
Crate:

- `datafusion/functions`

Primary file:

- `datafusion/functions/src/strings.rs`

Likely early adopters:

- `datafusion/functions/src/string/replace.rs`
- possibly one additional small UDF already using `try_append_with`, if needed to prove ergonomics

## Design Questions
1. Should the existing `StringWriter` trait remain infallible for current `append_with` callers?
2. Should `try_append_with` change signature, or should a new `try_append_with_checked` be added first?
3. Should the checked writer track the row-start length and total builder length, or only projected total builder length?
4. Should errors report the offset type limit only, or also include attempted output size?
5. Should checked writes reserve before each write or batch reservation for performance?

## Implementation Plan
1. Add a private checked writer type for `GenericStringArrayBuilder<O>`.
2. Add a small helper that validates `value_buffer.len() + additional_len <= O::MAX_OFFSET` before each write.
3. Implement checked write methods for `&str` and `char`.
4. Add a fallible append method that:
   - records `old_len`,
   - runs the closure with the checked writer,
   - rolls back on closure error,
   - pushes the final offset only on success.
5. Migrate `replace_into_writer` or add a checked variant:
   ```rust
   fn try_replace_into_writer<W: TryStringWriter>(...) -> Result<()> {
       ...
       w.try_write_str(...)?;
       ...
   }
   ```
6. Keep old infallible writer path for existing non-migrated callers.
7. Add targeted tests around mid-row overflow and rollback.

## Correctness Constraints
- A failed checked write must leave the builder in a valid state for later use.
- No failed checked append may push an offset.
- No failed checked append may increment placeholder count.
- Successful checked output must match current `append_with` / `try_append_with` output exactly.
- UTF-8 validity must still come from `&str` / `char` writes.
- Error text should remain compatible with the existing fallible builder overflow error:
  `"byte array offset overflow: output size exceeds {} bytes"`.

## Test Plan
Add builder-level unit tests in `datafusion/functions/src/strings.rs`:

- checked writer succeeds for multiple `try_write_str` and `try_write_char` calls,
- checked writer errors when a write would exceed `O::MAX_OFFSET`,
- failed checked append leaves previous successful rows readable after `finish`,
- failed checked append does not add an offset row,
- subsequent append after failed checked append still works.

Add one UDF-level regression if an early adopter is migrated:

```bash
cargo test -p datafusion-functions string::replace --lib
```

Run broader crate checks:

```bash
cargo test -p datafusion-functions strings::tests --lib
cargo test -p datafusion-functions --lib
cargo test -p datafusion-functions --lib --no-default-features
```

## Risks
- **API churn inside `datafusion/functions`:** changing `try_append_with` signature can force many call-site updates.
- **Performance risk:** per-write offset checks may add overhead in hot string UDFs.
- **Complexity risk:** supporting both infallible and fallible writer traits may temporarily increase abstraction count.
- **Rollback semantics:** closure errors, not just overflow errors, must leave the builder reusable.

## Performance Notes
This refactor adds checks to generated-output writes. Before migrating hot paths broadly, consider microbenchmarks or at least compare representative string UDF tests. If per-write checks are too expensive, keep both APIs:

- existing `try_append_with` for paths with cheap final-size validation,
- checked writer path for UDFs where generated output can grow substantially and early failure matters.

## Acceptance Criteria
- A checked writer abstraction exists for `GenericStringArrayBuilder` generated output.
- At least one caller demonstrates ergonomic use.
- Builder remains valid and reusable after checked writer failure.
- Existing `StringWriter` callers continue to work or are migrated in a reviewable way.
- Targeted tests cover success, overflow, rollback, and subsequent reuse.
- No broad unrelated UDF migrations are bundled into this refactor.
