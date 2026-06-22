source: refactor-byte-accounting-03-22688a
# Refactor: Consolidate `StringViewArrayBuilder` Capacity and Limit Checks

## Summary

Consolidate the capacity growth and ByteView limit checks inside `StringViewArrayBuilder` so fallible and infallible append paths share the same invariants for long string views.

## Background

`StringViewArrayBuilder` builds `StringViewArray` values using:

- inline views for strings up to 12 bytes
- an in-progress out-of-line data block for longer values
- completed blocks referenced by `ByteView.buffer_index`

Arrow `ByteView` stores several fields as signed 32-bit-compatible values. The builder must ensure these values remain within `i32::MAX`:

- value length
- offset into the current block
- completed buffer count / `buffer_index`

Recent work added fallible append APIs so some string UDF paths can return `DataFusionError` instead of panicking on these limits. That change introduced checked logic in `try_append_value`, while the older infallible paths still use adjacent but separate logic in `ensure_long_capacity`, `append_byte_map`, and `append_with`.

## Problem

`StringViewArrayBuilder::try_append_value` now has capacity-growth and limit-check logic that overlaps with `ensure_long_capacity` and the infallible append paths:

- `try_append_value` computes checked `required_cap`
- `ensure_long_capacity` computes similar capacity growth infallibly
- `append_byte_map` and `append_with` independently convert offsets and buffer counts with `expect`
- long-view construction uses both checked and infallible helper shapes

This duplication is behaviorally correct today, but it creates drift risk. Future changes to block growth, limit checks, error messages, or ByteView construction could update one path and miss another.

This is especially important because these helpers ultimately construct unsafe `StringViewArray` internals. The invariant should be encoded once at the builder layer, not repeated across append methods.

## Goals

- Preserve behavior exactly.
- Keep public interfaces unchanged.
- Centralize long-string capacity growth logic.
- Centralize ByteView `i32::MAX` limit checks for length, offset, and buffer index.
- Keep infallible methods panicking where they currently panic.
- Keep fallible methods returning `DataFusionError` with the existing `byte array offset overflow` wording.
- Make the code easier to audit for unsafe `StringViewArray` invariants.

## Non-Goals

- Do not change `StringViewArrayBuilder` public API shape.
- Do not change output arrays, null semantics, block layout, or performance characteristics intentionally.
- Do not migrate unrelated UDFs.
- Do not replace allocator/OOM behavior with fallible reserve handling unless separately scoped.
- Do not change `GenericStringArrayBuilder` behavior.

## Proposed Approach

### 1. Add a fallible capacity helper

Introduce a private helper that performs the checked equivalent of `ensure_long_capacity`:

```rust
fn try_ensure_long_capacity(&mut self, length: u32) -> Result<()> {
    let required_cap = self
        .in_progress
        .len()
        .checked_add(length as usize)
        .ok_or_else(|| string_view_overflow_error("string view block size"))?;

    if self.in_progress.capacity() < required_cap {
        self.flush_in_progress();
        let to_reserve = (length as usize).max(self.next_block_size() as usize);
        self.in_progress.reserve(to_reserve);
    }

    Ok(())
}
```

Then update `try_append_value` to call it instead of duplicating this logic.

### 2. Keep infallible helper behavior via wrapper

Keep `ensure_long_capacity` as the infallible API for existing infallible append paths, but make it delegate to the fallible helper:

```rust
fn ensure_long_capacity(&mut self, length: u32) {
    self.try_ensure_long_capacity(length)
        .expect("byte array offset overflow");
}
```

Confirm that panic behavior remains compatible with the current infallible methods.

### 3. Centralize checked ByteView field conversion

Consider small private helpers for repeated conversions:

```rust
fn try_i32_u32(value: usize, field: &str) -> Result<u32> {
    Ok(i32::try_from(value)
        .map_err(|_| string_view_overflow_error(field))? as u32)
}

fn i32_u32(value: usize, field: &str) -> u32 {
    i32::try_from(value)
        .unwrap_or_else(|_| panic!("{field} exceeds i32::MAX")) as u32
}
```

Use only if it improves readability. Avoid over-abstracting if the direct conversions remain clearer.

### 4. Keep long-view construction single-purpose

Preserve a checked/static helper for already-validated fields and a receiver helper for infallible paths if useful:

```rust
fn make_long_view_checked(
    length: u32,
    buffer_index: u32,
    offset: u32,
    prefix_bytes: &[u8],
) -> u128
```

Ensure callers do not duplicate buffer-index conversion unnecessarily.

## Files to Modify

- `datafusion/functions/src/strings.rs`

Likely affected methods:

- `StringViewArrayBuilder::try_append_value`
- `StringViewArrayBuilder::ensure_long_capacity`
- `StringViewArrayBuilder::append_byte_map`
- `StringViewArrayBuilder::append_with`
- private long-view construction helpers

## Testing Plan

Run existing builder tests first and after each refactor step:

```bash
cargo test -p datafusion-functions strings::tests --lib
```

Also run no-default-feature coverage for touched builder code:

```bash
cargo test -p datafusion-functions --lib --no-default-features strings::tests
```

Add or adjust tests only if the refactor exposes an untested invariant. Candidate test areas:

- long string append through `try_append_value`
- long string append through `append_byte_map`
- long string append through `append_with`
- block rotation / completed buffer references
- placeholder behavior remains unchanged

If practical without huge allocations, add a low-memory unit test for a checked overflow helper. If not practical, keep relying on existing direct helper tests and document why true StringView limit overflow is not exercised.

## Acceptance Criteria

- `try_append_value` and `ensure_long_capacity` no longer duplicate capacity-growth logic.
- Fallible paths still return `DataFusionError` with `byte array offset overflow` in limit failures.
- Infallible paths still panic on the same limit failures.
- Existing output arrays and block reuse behavior are unchanged.
- Existing `strings::tests` pass.
- No public API changes.
- No unrelated UDF changes.

## Risk Assessment

- **Correctness risk:** Medium. These helpers build unsafe `StringViewArray` internals, so field conversion and block references must remain exact.
- **Behavior risk:** Low if tests confirm identical outputs and panic/error modes.
- **Performance risk:** Low. The refactor should only move existing checks; avoid adding allocations or changing hot-loop behavior.
- **Review risk:** Low-to-medium. Keep the patch small and local to `StringViewArrayBuilder`.

## Suggested Implementation Notes

- Preserve the existing `i32::try_from(... ) as u32` semantics. Do not replace with `u32::try_from`, because values above `i32::MAX` must remain invalid for Arrow ByteView compatibility.
- Keep comments focused on the ByteView signed-32-bit invariant, not on mechanical casts.
- Avoid broad trait abstractions. Private helpers are enough.
- Prefer one small refactor commit with no behavior changes.
