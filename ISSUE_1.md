# Eliminate positional fallback in struct casting — require at least one matching field name

## Problem

Currently, DataFusion's struct casting includes a **positional fallback** mechanism that allows casting between structs with completely different field names if the field counts match. This violates DuckDB's semantics and can silently corrupt data.

**Example of problematic behavior:**
```rust
// This currently succeeds but should fail
source: struct<left int, right varchar>
target: struct<alpha int, beta varchar>

// Positional fallback causes:
// left → alpha, right → beta
// Despite zero matching field names!
```

## Current Behavior

The implementation in [`validate_struct_compatibility()`](datafusion/common/src/nested_struct.rs) currently allows casting when:
- No field name overlap exists AND
- Field counts match

The casting correctly fails only when there's no overlap **and** counts differ with the error:
```
Cannot cast struct with X fields to Y fields without name overlap; positional mapping is ambiguous
```

## Proposed Solution

Align with **DuckDB's semantics** by requiring at least one matching field name for any struct cast to succeed:

1. **Require name-based matching** — at least one field name must match between source and target
2. **Remove positional fallback** — even when field counts are equal
3. **Clear error message** when no field names match

**Implementation approach:**

Update the validation logic in `validate_struct_compatibility()` to reject casts without field name overlap:

```rust
let has_overlap = fields_have_name_overlap(source_fields, target_fields);
if !has_overlap {
    // DuckDB semantics: require at least one matching field
    return _plan_err!(
        "Cannot cast struct: at least one field name must match between source and target. \
         Source fields: {:?}, Target fields: {:?}",
        source_fields.iter().map(|f| f.name()).collect::<Vec<_>>(),
        target_fields.iter().map(|f| f.name()).collect::<Vec<_>>()
    );
}
```

## Impact

- **Breaking change** — Any code relying on positional casting with no name overlap will fail
- **Safety improvement** — Prevents silent data corruption from accidental field misalignment
- **Consistency** — Aligns with DuckDB's well-designed struct casting semantics and the principle of "at least one field name must match"

## Acceptance Criteria

- [ ] Remove positional fallback code path in `cast_struct_column()`
- [ ] Update validation in `validate_struct_compatibility()` to reject casts without name overlap
- [ ] Add clear error message: "Cannot cast struct: at least one field name must match between source and target"
- [ ] Update existing tests that relied on positional fallback behavior
- [ ] Add test case verifying that `struct<a, b> → struct<x, y>` correctly fails
- [ ] Verify all struct casting tests pass
- [ ] Update documentation/comments to clarify DuckDB-aligned behavior

## Related Issues

- Companion issue: Validate non-nullable missing fields in struct casting (should be implemented separately)
- Related PR: struct-casting-17285b

## Context

This issue stems from a comprehensive review of struct casting semantics to align with DuckDB. The current PR already handles partial overlap correctly (e.g., `struct<c1, other1> → struct<other2, c1>` correctly matches on `c1`), but the positional fallback for zero-overlap cases needs to be removed.

The complete analysis and rationale is available in `PR_RESPONSE.md` (section "Response to: No Positional Casting").
