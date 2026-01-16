# PR Review Response: Struct Casting Semantics

## Response to: Partial Overlap Case

**Comment:**
> What if there is partial overlap, e.g.:
> ```
> struct<c1 int, other1 int> -> struct<other2 int, c1 int>
> ```
> I think we should follow **DuckDB's semantics**, unless there is enough agreement between other data systems that support structs:
> 
> **When casting between structs, the names of at least one field have to match.**

**Response:**

Thank you for raising this important edge case. The current implementation **already handles partial overlap correctly** by using name-based matching when any names overlap. 

For your example `struct<c1 int, other1 int> -> struct<other2 int, c1 int>`:

- **Current behavior:** The cast succeeds because `c1` matches
- Field `c1` from source is cast to target field `c1` (preserving the value)
- Target field `other2` has no match in source → filled with NULL
- Source field `other1` has no match in target → ignored

This aligns perfectly with DuckDB's requirement that "at least one field name must match."

### Implementation Details

The name-overlap detection is handled by the `fields_have_name_overlap()` function in [nested_struct.rs](datafusion/common/src/nested_struct.rs#L341-L348):

```rust
fn fields_have_name_overlap(
    source_fields: &[FieldRef],
    target_fields: &[FieldRef],
) -> bool {
    let source_names: HashSet<&str> = source_fields
        .iter()
        .map(|field| field.name().as_str())
        .collect();
    target_fields
        .iter()
        .any(|field| source_names.contains(field.name().as_str()))
}
```

When `has_overlap` is true, the casting logic uses name-based matching for **all** fields, not just the overlapping ones. This means:
- Matching fields are cast by name
- Missing target fields are filled with NULLs
- Extra source fields are dropped

---

## Response to: No Positional Casting

**Comment:**
> Thus, there is **no positional casting**. Casting succeeds if:
> * There is **at least one matching field** that can be cast.
> * **All matching fields** can be cast successfully.
> * Any **non-nullable target fields** match with a **non-nullable source field**.

**Response:**

I appreciate the clarity of DuckDB's approach. The current implementation includes **positional fallback** as a pragmatic compromise, but I agree that aligning more closely with DuckDB would improve consistency and safety.

### Current Implementation

The PR currently includes positional fallback in this specific scenario:
- **No name overlap** (zero matching field names) AND
- **Field counts match**

Example: `struct<left int, right varchar> -> struct<alpha int, beta varchar>`
- Since there are no matching names but counts match (2 = 2), this falls back to positional casting
- `left` → `alpha`, `right` → `beta`

When there's **no overlap and counts differ**, the cast correctly fails with:
```
Cannot cast struct with X fields to Y fields without name overlap; positional mapping is ambiguous
```

### Recommendation: Eliminate Positional Fallback

**Alignment with DuckDB's semantics would mean:**

1. **Require at least one matching field name** for any struct cast to succeed
2. **No positional fallback** - even when field counts match
3. This prevents accidental data corruption when field names are completely different

**Proposed change:** Update the validation logic in `validate_struct_compatibility()` to:

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

**Impact:** This would be a **breaking change** for any code relying on positional casting when there's no name overlap. However, it's a safer default that prevents silent data corruption.

---

## Response to: Recommendations for Aligning with DuckDB

### 1. Require at least one matching field - Eliminate positional fallback entirely

**Status:** ✅ **Agreed - should be implemented**

As discussed above, removing the positional fallback entirely would improve safety and align with DuckDB. This is a reasonable breaking change that prevents subtle bugs.

**Action Item:**
- Remove the positional fallback code path in `cast_struct_column()`
- Update validation to require at least one name match
- Update tests and documentation

---

### 2. Consider case-insensitive matching - DuckDB treats x and X as matching

**Status:** ⚠️ **Requires broader discussion**

**Current behavior:** Case-sensitive matching (e.g., `x` and `X` are different fields)

**DuckDB behavior:** Case-insensitive matching (e.g., `x` and `X` match)

**Considerations:**

**Pros of case-insensitive:**
- Aligns with DuckDB
- More forgiving for common casing variations (e.g., JSON field name inconsistencies)
- Follows SQL's general case-insensitivity for identifiers

**Pros of case-sensitive (current):**
- Matches Arrow's field name handling (Arrow is case-sensitive):
  - [`Fields::find()`](https://github.com/apache/arrow-rs/blob/main/arrow-schema/src/fields.rs#L83) uses exact string equality: `b.name() == name`
  - [`StructArray::column_by_name()`](https://github.com/apache/arrow-rs/blob/main/arrow-array/src/array/struct_array.rs#L324) compares with `==`: `c == &column_name`
  - [`Field` equality](https://github.com/apache/arrow-rs/blob/main/arrow-schema/src/field.rs#L109) uses direct comparison: `self.name == other.name`
- Consistent with Rust and JSON conventions
- Prevents ambiguity when source has both `x` and `X` (though this is rare)
- More predictable behavior for programmatic use

**Recommendation:** 

I suggest **keeping case-sensitive matching** for now because:
1. Arrow (DataFusion's foundation) is case-sensitive
2. It's more conservative and prevents ambiguous matches
3. Users can explicitly cast or rename fields if case differs
4. We can always relax to case-insensitive later if needed (harder to go the other direction)

If we want to align with DuckDB on this, it should be a deliberate design decision with community input, as it affects the broader type system.

---

### 3. Add validation for missing non-nullable fields

**Comment:**
> If target field is non-nullable but missing from source, error instead of filling with NULL

**Status:** ⚠️ **Partially implemented, needs enhancement**

**Current behavior:**

The PR already validates nullability **for matching fields**:

```rust
// In validate_field_compatibility()
if source_field.is_nullable() && !target_field.is_nullable() {
    return _plan_err!(
        "Cannot cast nullable struct field '{}' to non-nullable field",
        target_field.name()
    );
}
```

However, when a target field is **missing from the source**, the current implementation fills it with NULL regardless of the target field's nullability constraint.

**Issue:** If target has a non-nullable field that doesn't exist in source, we incorrectly fill it with NULL.

**Example that should fail but currently succeeds:**
```sql
-- Source: {a: 1}
-- Target: STRUCT(a INT, b INT NOT NULL)
-- Currently: succeeds and fills b with NULL
-- Should: error because b is NOT NULL but missing from source
```

**Recommended fix:** Add validation in `validate_struct_compatibility()`:

```rust
// After checking matching fields
for target_field in target_fields {
    if let Some(source_field) = source_fields
        .iter()
        .find(|f| f.name() == target_field.name())
    {
        // existing field compatibility check
        validate_field_compatibility(source_field, target_field)?;
    } else {
        // Target field is missing from source
        if !target_field.is_nullable() {
            return _plan_err!(
                "Cannot cast struct: target field '{}' is non-nullable but missing from source. \
                 Cannot fill with NULL.",
                target_field.name()
            );
        }
    }
}
```

**Action Item:**
- Add validation for missing non-nullable target fields
- Add test case for this scenario
- Update error messages to be more descriptive

---

## Summary of Recommendations

| Recommendation | Status | Priority | Breaking Change |
|---------------|--------|----------|-----------------|
| **1. Eliminate positional fallback** | ✅ Agreed | **High** | Yes - safer default |
| **2. Case-insensitive matching** | ⚠️ Needs discussion | Low | Yes - type system impact |
| **3. Validate non-nullable missing fields** | ✅ Should implement | **High** | Yes - correctness fix |

### Proposed Follow-up Changes

If we agree on recommendations 1 and 3, I can prepare a follow-up commit to:

1. Remove the positional fallback entirely
2. Add validation for missing non-nullable target fields  
3. Update tests to reflect the stricter semantics
4. Update documentation/comments to clarify the DuckDB-aligned behavior

This would make DataFusion's struct casting semantics:
- ✅ **Safer** (prevents silent data corruption)
- ✅ **More compatible** with DuckDB
- ✅ **More predictable** (single clear rule: match by name, require at least one match)

### Case-Insensitive Matching Discussion

The case-insensitive matching (recommendation 2) deserves a separate discussion/issue as it has broader implications:
- Should DataFusion generally follow SQL's case-insensitivity or stay case-sensitive like Arrow?
- How would this interact with other parts of the system (identifier resolution, schema merging, etc.)?
- What's the performance impact of case-insensitive comparisons?

I recommend we address recommendations 1 and 3 first (which are clear improvements) and defer the case-sensitivity decision to a broader architectural discussion.

---

## Conclusion

Thank you for the detailed comparison with DuckDB! The suggestions align well with DataFusion's goals for correctness and compatibility. The current PR already handles partial overlap correctly, but eliminating the positional fallback and adding stricter nullability validation would make the implementation more robust and aligned with DuckDB's well-designed semantics.

Please let me know if you'd like me to prepare a follow-up commit implementing recommendations 1 and 3, or if you'd prefer to discuss the approach further.
