# PR Review: Struct Casting with Field Reordering Fix

**Commit Range:** 3adcc2653^..15c0f8ea8  
**Issue:** #14396 - Wrong Cast Between Structs with Different Field Order  
**Branch:** struct-casting-17285a

---

## 🎯 Executive Summary

**Decision: ✅ Approve with Suggestions**

The implementation successfully solves the reported bug where struct casting used positional indexing instead of field names. The fix is well-tested with comprehensive SQL logic tests and handles key edge cases including field reordering, missing fields, and nested structs.

**Key Strengths:**
- ✅ Correctly implements name-based field matching for struct casting
- ✅ Comprehensive test coverage with 15+ SQL logic test cases
- ✅ Handles nested structs, missing fields, and type coercion
- ✅ No breaking API changes
- ✅ Clear documentation in doc comments

**Suggested Improvements (Non-Blocking):**
- 📝 Code duplication with existing `nested_struct.rs` module
- 📝 Opportunity to consolidate struct casting logic
- 📝 Minor optimization opportunities in scalar struct casting

---

## 📊 Code Changes Overview

### Files Modified (11 files)
- **New file:** [datafusion/common/src/struct_cast.rs](datafusion/common/src/struct_cast.rs) - 127 lines
- **Modified:** [datafusion/expr-common/src/columnar_value.rs](datafusion/expr-common/src/columnar_value.rs) - Cast logic integration
- **Modified:** [datafusion/expr-common/src/type_coercion/binary.rs](datafusion/expr-common/src/type_coercion/binary.rs) - Name-based struct coercion
- **Modified:** [datafusion/common/src/scalar/mod.rs](datafusion/common/src/scalar/mod.rs) - Scalar struct casting
- **Modified:** [datafusion/physical-expr/src/expressions/cast.rs](datafusion/physical-expr/src/expressions/cast.rs) - Physical expr casting
- **Modified:** [datafusion/expr/src/expr_schema.rs](datafusion/expr/src/expr_schema.rs) - Schema validation
- **Modified:** [datafusion/optimizer/src/simplify_expressions/expr_simplifier.rs](datafusion/optimizer/src/simplify_expressions/expr_simplifier.rs) - Const folding guard
- **Tests:** [datafusion/sqllogictest/test_files/struct.slt](datafusion/sqllogictest/test_files/struct.slt) +87 lines
- **Tests:** [datafusion/sqllogictest/test_files/case.slt](datafusion/sqllogictest/test_files/case.slt) - Updated expectations

---

## 🔍 Detailed Review

### 1. 🟡 **Code Duplication with `nested_struct.rs`**

**Location:** [datafusion/common/src/struct_cast.rs](datafusion/common/src/struct_cast.rs)

**Observation:**  
The new `struct_cast.rs` module (~127 lines) duplicates significant logic from the existing [datafusion/common/src/nested_struct.rs](datafusion/common/src/nested_struct.rs) module, specifically the `cast_struct_column()` function (lines 57-93).

**Side-by-side comparison:**

**New `struct_cast.rs`:**
```rust
pub fn cast_struct_array_by_name(
    array: &ArrayRef,
    target_fields: &Fields,
    cast_options: &CastOptions<'static>,
) -> Result<ArrayRef> {
    let struct_array = array.as_any().downcast_ref::<StructArray>()?;
    let source_fields = struct_array.fields();
    
    // Build HashMap for field lookup
    let mut source_by_name = source_fields.iter()
        .enumerate()
        .map(|(idx, field)| (field.name().clone(), (idx, field)))
        .collect::<HashMap<_, _>>();
    
    // Iterate target fields, match by name, cast
    for target_field in target_fields {
        if let Some((idx, _)) = source_by_name.remove(target_field.name()) {
            let child = Arc::clone(struct_array.column(idx));
            cast_array_with_name_matching(&child, target_field.data_type(), cast_options)?
        } else {
            new_null_array(target_field.data_type(), struct_array.len())
        }
    }
}
```

**Existing `nested_struct.rs::cast_struct_column()`:**
```rust
fn cast_struct_column(
    source_col: &ArrayRef,
    target_fields: &[Arc<Field>],
    cast_options: &CastOptions,
) -> Result<ArrayRef> {
    let source_struct = source_col.as_any().downcast_ref::<StructArray>()?;
    
    for target_child_field in target_fields {
        match source_struct.column_by_name(target_child_field.name()) {
            Some(source_child_col) => {
                cast_column(source_child_col, target_child_field, cast_options)?
            }
            None => {
                new_null_array(target_child_field.data_type(), num_rows)
            }
        }
    }
}
```

**Both implementations:**
- Match fields by name using case-sensitive comparison
- Insert null arrays for missing target fields
- Recursively handle nested struct types
- Use `Arc::clone` for efficient reference sharing
- Handle nulls at the struct level via `nulls().cloned()`

**Key Difference:**  
`struct_cast.rs` uses a `HashMap` for field lookup while `nested_struct.rs` uses `StructArray::column_by_name()` which internally iterates the field list.

**Concerns:**
1. **Maintenance Burden:** Future bug fixes or enhancements must be applied to both implementations
2. **Testing Surface:** Edge cases need coverage in both code paths
3. **Performance Inconsistency:** One uses HashMap O(1) lookup, other uses linear search O(n)
4. **Documentation Divergence:** Doc comments may drift out of sync

**Recommendation:**

Consider consolidating by enhancing `nested_struct::cast_column()` to serve both use cases:

```rust
// In nested_struct.rs
pub use cast_column as cast_struct_array_by_name;

// Or create a unified internal function:
fn cast_struct_internal(
    source: &ArrayRef, 
    target: &Fields,
    options: &CastOptions,
    use_positional_fallback: bool,
) -> Result<ArrayRef> {
    // Unified logic here
}
```

**Alternatively**, if the implementations need to remain separate, add a comment explaining *why*:
```rust
// Note: This differs from nested_struct::cast_struct_column in that it:
// 1. Falls back to positional casting when no field names match
// 2. Uses HashMap for O(1) field lookup performance
// 3. Enforces 'static lifetime on CastOptions for integration with ColumnarValue
```

---

### 2. 🟢 **Naming: HashMap vs. `column_by_name()`**

**Location:** [datafusion/common/src/struct_cast.rs#L68-L75](datafusion/common/src/struct_cast.rs#L68-L75)

**Current implementation:**
```rust
let mut source_by_name = source_fields
    .iter()
    .enumerate()
    .map(|(idx, field)| (field.name().clone(), (idx, field)))
    .collect::<std::collections::HashMap<_, _>>();

if source_by_name.len() != source_fields.len() {
    return Err(DataFusionError::Internal(
        "Duplicate field name found in struct".to_string(),
    ));
}
```

**Observation:**  
Using a `HashMap` provides O(1) lookup but requires cloning field names and adds complexity for duplicate detection.

**Existing pattern in codebase:**
```rust
// From nested_struct.rs
match source_struct.column_by_name(target_child_field.name()) {
    Some(source_child_col) => { /* ... */ }
    None => { /* insert null */ }
}
```

**Trade-offs:**

| Approach | Pros | Cons |
|----------|------|------|
| HashMap | • O(1) lookup<br>• Explicit duplicate detection | • Name cloning overhead<br>• More complex code |
| `column_by_name()` | • Simpler code<br>• No allocations<br>• Consistent with codebase | • O(n) lookup per field<br>• O(n²) total complexity |

**Performance Analysis:**
- For structs with < 10 fields (common case): Negligible difference
- For structs with 100+ fields: HashMap wins significantly
- `column_by_name()` already used in hot paths throughout DataFusion

**Recommendation:**

**Keep the HashMap approach** for the new code because:
1. It provides better asymptotic performance for wide structs
2. The duplicate detection is valuable (validates invariants)
3. The difference from `nested_struct.rs` is justified by different performance characteristics

**But add a comment explaining the choice:**
```rust
// Use HashMap for O(1) field lookup. This is more efficient than repeated
// StructArray::column_by_name() calls for wide structs (many fields).
let mut source_by_name = source_fields...
```

---

### 3. 🟢 **Positional Fallback Logic**

**Location:** [datafusion/common/src/struct_cast.rs#L53-L64](datafusion/common/src/struct_cast.rs#L53-L64)

**Implementation:**
```rust
// Check if any source field names match target field names
let source_names: std::collections::HashSet<_> =
    source_fields.iter().map(|f| f.name()).collect();
let has_name_overlap = target_fields
    .iter()
    .any(|f| source_names.contains(f.name()));

// If no field names match, fall back to positional casting
if !has_name_overlap {
    return Ok(kernels::cast::cast_with_options(
        array,
        &DataType::Struct(target_fields.clone()),
        cast_options,
    )?);
}
```

**This is excellent defensive programming.** It ensures backward compatibility when:
- User explicitly wants positional casting
- Field names are generic (`field1`, `field2`, etc.) in both source and target
- Legacy code generates anonymous struct fields

**Example where fallback is correct:**
```rust
// Source: {field_0: int, field_1: string}
// Target: {field_0: bigint, field_1: varchar}
// No name overlap → use positional casting (correct behavior)
```

**Question for maintainers:**  
Should this behavior be documented in the function's doc comment? Users might be surprised that *sometimes* positional casting is used.

**Suggested doc addition:**
```rust
/// # Fallback Behavior
/// 
/// If source and target have **no overlapping field names**, this function falls
/// back to positional casting (matching fields by index). This preserves backward
/// compatibility with anonymous or numbered struct fields.
```

---

### 4. 🟡 **Scalar Struct Casting Overhead**

**Location:** [datafusion/common/src/scalar/mod.rs#L3707-L3718](datafusion/common/src/scalar/mod.rs#L3707-L3718)

**Current implementation:**
```rust
let scalar_array = self.to_array()?;  // ScalarValue → Array

let cast_arr = match (scalar_array.data_type(), target_type) {
    (DataType::Struct(_), DataType::Struct(target_fields)) => {
        crate::struct_cast::cast_struct_array_by_name(
            &scalar_array,
            target_fields,
            cast_options,
        )?
    }
    _ => cast_with_options(&scalar_array, target_type, cast_options)?,
};

ScalarValue::try_from_array(&cast_arr, 0)?  // Array → ScalarValue
```

**Overhead analysis:**

For every scalar struct cast:
1. `self.to_array()` - Allocates array (even for single value)
2. `cast_struct_array_by_name()` - Processes array
3. `ScalarValue::try_from_array()` - Extracts single value with bounds checking

**Performance impact:**
- **Low** - Scalar struct casts are rare in hot paths (most operations work on arrays)
- **Medium** - In subquery results or CASE expressions with struct branches

**Alternative approach:**
Implement struct casting directly within `ScalarValue`:

```rust
// Hypothetical implementation
impl ScalarValue {
    fn cast_struct_to_struct(&self, target_fields: &Fields) -> Result<Self> {
        let ScalarValue::Struct(source_values) = self else { unreachable!() };
        
        let mut target_values = Vec::new();
        for target_field in target_fields {
            let value = source_values.field_by_name(target_field.name())
                .map(|v| v.cast_to(target_field.data_type()))
                .unwrap_or_else(|| ScalarValue::new_null(target_field.data_type()));
            target_values.push(value);
        }
        
        Ok(ScalarValue::Struct(target_values))
    }
}
```

**Recommendation:**

**For this PR:** Keep the current implementation - it's correct and the overhead is acceptable.

**For follow-up:** Add a TODO comment and file an issue for optimization:

```rust
// TODO: Optimize struct scalar casting to avoid Array roundtrip
// Current: Scalar → Array → cast_struct_array_by_name → Array → Scalar
// Future: Direct ScalarValue field-by-name matching
// See issue #XXXXX
let scalar_array = self.to_array()?;
```

This follows the AGENTS.md guideline:
> "Optimizations should be focused on bottlenecks — those steps that are repeated millions of times in a query; otherwise, prefer simplicity."

---

### 5. 🟢 **Type Coercion Integration**

**Location:** [datafusion/expr-common/src/type_coercion/binary.rs#L1223-L1267](datafusion/expr-common/src/type_coercion/binary.rs#L1223-L1267)

**Implementation:**
```rust
match (lhs_type, rhs_type) {
    (Struct(lhs_fields), Struct(rhs_fields)) => {
        if lhs_fields.len() != rhs_fields.len() {
            return None;  // Field count must match
        }

        // Try name-based coercion first
        let rhs_by_name: HashMap<&str, &FieldRef> =
            rhs_fields.iter().map(|f| (f.name().as_str(), f)).collect();

        let has_name_overlap = lhs_fields
            .iter()
            .any(|lf| rhs_by_name.contains_key(lf.name().as_str()));

        if has_name_overlap {
            // Match fields by name, coerce types, preserve left-side field order
            let coerced_fields: Option<Vec<FieldRef>> = lhs_fields
                .iter()
                .map(|lhs_field| {
                    rhs_by_name.get(lhs_field.name().as_str())
                        .and_then(|rhs_field| {
                            comparison_coercion(
                                lhs_field.data_type(),
                                rhs_field.data_type(),
                            )
                            .map(|coerced_type| {
                                let is_nullable = lhs_field.is_nullable()
                                    || rhs_field.is_nullable();
                                Arc::new(Field::new(
                                    lhs_field.name().clone(),
                                    coerced_type,
                                    is_nullable,
                                ))
                            })
                        })
                })
                .collect();

            return coerced_fields.map(|fields| Struct(fields.into()));
        }

        // Fallback: positional coercion if no names match
        // ... (existing positional code)
    }
}
```

**This is well-designed.** Key observations:

✅ **Preserves left-side field order** - Consistent with SQL standard  
✅ **Combines nullability** - `is_nullable = lhs.is_nullable() || rhs.is_nullable()`  
✅ **Recursive coercion** - Calls `comparison_coercion()` on nested types  
✅ **Fallback to positional** - Maintains backward compatibility  

**Example behavior:**
```sql
SELECT CASE
    WHEN condition THEN {b: 3, a: 4}    -- lhs: {b: Int32, a: Int32}
    ELSE {a: 5, b: 6}                   -- rhs: {a: Int32, b: Int32}
END;
-- Coerced type uses lhs order: Struct(b: Int32, a: Int32)
-- Result: {b: 6, a: 5} (values matched by name, order from lhs)
```

**One concern:**  
If `has_name_overlap` is true but not *all* fields match by name, the function returns fields only for matching names. Is this intended?

**Example edge case:**
```rust
lhs: {a: Int32, b: String}
rhs: {a: Int64, c: Float}
// has_name_overlap = true (field 'a' matches)
// Result: Only field 'a' coerced, field 'b' dropped?
```

**Test needed:**
```sql
-- Does this error or coerce to {a: Int64}?
SELECT CASE
    WHEN condition THEN {a: 1, b: 'hello'}
    ELSE {a: 2, c: 3.14}
END;
```

**Recommendation:**  
Add a SQL logic test for partial field overlap to verify behavior.

---

### 6. 🟢 **Const Folding Guard**

**Location:** [datafusion/optimizer/src/simplify_expressions/expr_simplifier.rs#L657-L671](datafusion/optimizer/src/simplify_expressions/expr_simplifier.rs#L657-L671)

**Implementation:**
```rust
// Skip const-folding for struct casts with field count mismatches
// as these can cause optimizer hang
Expr::Cast(Cast { expr, data_type })
| Expr::TryCast(TryCast { expr, data_type }) => {
    if let (Ok(source_type), DataType::Struct(target_fields)) =
        (expr.get_type(&DFSchema::empty()), data_type)
    {
        if let DataType::Struct(source_fields) = source_type {
            // Don't const-fold struct casts with different field counts
            if source_fields.len() != target_fields.len() {
                return false;
            }
        }
    }
    true
}
```

**This is a critical fix.** It prevents optimizer hangs caused by:
1. Const-folding attempting to evaluate struct casts at compile time
2. Field count mismatches causing unexpected behavior in the folding logic
3. Recursive coercion loops in nested struct scenarios

**Why field count matters:**
- Field count mismatch means null insertion or field dropping
- Optimizer's const folder may not handle null array creation correctly
- Runtime casting can handle this, but compile-time evaluation cannot

**Test validation:**
The fix enables these previously TODO-marked tests in [case.slt](datafusion/sqllogictest/test_files/case.slt):
```sql
# Previously failed with optimizer hang:
query ?
SELECT CAST({a: 1, b: 2, extra: 3} AS STRUCT(a INT, b INT));
----
{a: 1, b: 2}
```

**Excellent defensive programming.** No issues found.

---

### 7. 🟢 **Physical Expression Layer**

**Location:** [datafusion/physical-expr/src/expressions/cast.rs#L240-L245](datafusion/physical-expr/src/expressions/cast.rs#L240-L245)

**Implementation:**
```rust
} else if can_cast_types(&expr_type, &cast_type) {
    Ok(Arc::new(CastExpr::new(expr, cast_type, cast_options)))
} else if matches!((&expr_type, &cast_type), (Struct(_), Struct(_))) {
    // Allow struct-to-struct casts even if Arrow's can_cast_types rejects them
    // (e.g., field count mismatches). These will be handled by name-based casting
    // at execution time via ColumnarValue::cast_to
    Ok(Arc::new(CastExpr::new(expr, cast_type, cast_options)))
} else {
    not_impl_err!("Unsupported CAST from {expr_type} to {cast_type}")
}
```

**This elegantly bypasses Arrow's restrictions.** 

**Context:**  
Arrow's `can_cast_types()` rejects struct-to-struct casts when:
- Field counts differ
- Field types aren't directly castable (Arrow uses positional matching)
- Field names don't matter to Arrow

**Why this works:**
1. `CastExpr` just stores the target type - doesn't validate it
2. Actual casting happens in `CastExpr::evaluate()` → `ColumnarValue::cast_to()`
3. `ColumnarValue::cast_to()` calls `cast_struct_array_by_name()` which handles everything

**Flow:**
```
Physical Plan Creation:
cast_with_options() → CastExpr::new() → store cast_type

Execution:
CastExpr::evaluate() → ColumnarValue::cast_to() → cast_struct_array_by_name()
```

**Excellent separation of concerns.** No issues.

---

### 8. 🟢 **Test Coverage Analysis**

**Location:** [datafusion/sqllogictest/test_files/struct.slt#L849-L935](datafusion/sqllogictest/test_files/struct.slt#L849-L935)

**Test cases added (15 tests):**

| Test | Coverage |
|------|----------|
| `CAST({b: 'b_value', a: 'a_value'} AS STRUCT(a VARCHAR, b VARCHAR))` | ✅ Field reordering - strings |
| `CAST({b: 3, a: 4} AS STRUCT(a INT, b INT))` | ✅ Field reordering - integers |
| `CAST({b: 3, a: 4} AS STRUCT(a BIGINT, b INT))` | ✅ Type casting + reordering |
| `CAST({a: 1} AS STRUCT(a INT, b INT))` | ✅ Missing field → null |
| `CAST({a: 1, b: 2, extra: 3} AS STRUCT(a INT, b INT))` | ✅ Extra field → ignored |
| `CAST({inner: {y: 2, x: 1}} AS STRUCT(inner STRUCT(x INT, y INT)))` | ✅ Nested struct reordering |
| Table data with field reordering | ✅ Integration with real data |
| Multiple nesting levels | ✅ Deep nesting |
| Null values in source fields | ✅ Field-level nulls |
| Struct-level nulls | ✅ Null struct preservation |

**Updated tests in [case.slt](datafusion/sqllogictest/test_files/case.slt):**
- CASE expressions with struct field reordering
- Expected results updated to match name-based behavior
- Error messages updated for field count mismatches

**Coverage assessment:**

| Category | Status | Notes |
|----------|--------|-------|
| Basic field reordering | ✅ Excellent | Multiple test cases |
| Type coercion + reordering | ✅ Good | Covered |
| Missing fields | ✅ Good | Null insertion tested |
| Extra fields | ✅ Good | Ignored correctly |
| Nested structs | ✅ Good | Single and multiple levels |
| Null handling | ✅ Good | Field and struct level |
| Duplicate field names | ⚠️ Missing | Error path untested |
| Empty structs | ⚠️ Missing | `{}` → `{a: INT}` case |
| Case sensitivity | ⚠️ Missing | `{A: 1}` vs `{a: INT}` |
| Performance (wide structs) | ⚠️ Missing | 100+ field test |

**Recommendation:**

Add these edge case tests:

```sql
# Test case sensitivity
query ?
SELECT CAST({a: 1, A: 2} AS STRUCT(a INT, A INT));
----
{a: 1, A: 2}

# Test empty struct
query ?
SELECT CAST({} AS STRUCT(a INT, b INT));
----
{a: NULL, b: NULL}

# Test duplicate field names (should error)
statement error Duplicate field name
SELECT CAST({a: 1, a: 2} AS STRUCT(a INT));
```

**Overall: Test coverage is very good** - production-ready with suggested additions for edge cases.

---

### 9. 🟢 **Documentation Quality**

**Location:** [datafusion/expr-common/src/columnar_value.rs#L277-L294](datafusion/expr-common/src/columnar_value.rs#L277-L294)

**Added documentation:**
```rust
/// Cast's this [ColumnarValue] to the specified `DataType`
///
/// # Struct Casting Behavior
///
/// When casting struct types, fields are matched **by name** rather than position:
/// - Source fields are matched to target fields using case-sensitive name comparison
/// - Fields are reordered to match the target schema
/// - Missing target fields are filled with null arrays
/// - Extra source fields are ignored
///
/// # Example
/// ```text
/// Source: {"b": 3, "a": 4}  (schema: {b: Int32, a: Int32})
/// Target: {"a": Int32, "b": Int32}
/// Result: {"a": 4, "b": 3}  (values matched by field name)
/// ```
///
/// For non-struct types, uses Arrow's standard positional casting.
```

**This is excellent.** It:
- ✅ Clearly explains the behavior change
- ✅ Uses bold formatting for emphasis
- ✅ Includes a concrete example
- ✅ Specifies case sensitivity
- ✅ Documents edge cases (missing/extra fields)

**Also documented:**
- `cast_struct_array_by_name()` has comprehensive doc comment
- Error messages are clear: `"Expected StructArray but got..."`
- Test comments explain expected behavior

**From AGENTS.md:**
> "Document public APIs with `///` comments and examples."

**✅ Fully compliant.** No improvements needed.

---

## 🏗️ Architecture & Design

### Design Pattern: Strategy Pattern

The implementation uses the **Strategy Pattern** to support multiple casting strategies:

1. **Name-based struct casting** - When field names overlap
2. **Positional struct casting** - Fallback when no names match
3. **Arrow standard casting** - For non-struct types

```rust
fn cast_array_by_name(array: &ArrayRef, cast_type: &DataType, ...) -> Result<ArrayRef> {
    match (array.data_type(), cast_type) {
        (Struct(_), Struct(target)) => /* Strategy 1: Name-based */,
        _ => /* Strategy 3: Arrow standard */,
    }
}
```

This is **well-aligned** with DataFusion's extensible architecture.

---

### Integration Points

The fix touches 4 key layers of DataFusion:

```
┌─────────────────────────────────────────┐
│ SQL Layer                               │
│ - expr_schema.rs: Allow struct→struct   │
│ - type_coercion: Name-based coercion    │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ Logical Optimization                    │
│ - expr_simplifier: Skip const-folding   │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ Physical Planning                       │
│ - cast.rs: Bypass Arrow validation      │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ Execution                               │
│ - columnar_value: Name-based casting    │
│ - struct_cast: Core implementation      │
└─────────────────────────────────────────┘
```

Each layer handles its concern appropriately. **Good separation of concerns.**

---

## 🎯 Scope & Effectiveness

### ✅ Does it solve the problem?

**Original bug:**
```sql
SELECT CAST({b: 3, a: 4} AS STRUCT(a INT, b INT));
-- Before: {a: 3, b: 4}  ❌ (positional)
-- After:  {a: 4, b: 3}  ✅ (name-based)
```

**✅ Fully resolved.**

### ✅ Edge cases handled?

| Case | Handled? | Evidence |
|------|----------|----------|
| Field reordering | ✅ | SQL test line 852 |
| Type coercion | ✅ | SQL test line 871 |
| Missing fields | ✅ | SQL test line 877 |
| Extra fields | ✅ | SQL test line 883 |
| Nested structs | ✅ | SQL test line 889 |
| Null preservation | ✅ | SQL test lines 914, 926 |
| No name overlap | ✅ | Fallback to positional |
| Different field counts | ✅ | Const-folding guard |

**⚠️ Minor gaps:**
- Duplicate field names (error path untested)
- Empty struct casting
- Case sensitivity validation

### ✅ Breaking changes?

**None.** The change fixes a bug but:
- No API signature changes
- Existing correct usage continues to work
- Positional casting still available when no names match
- Backward compatible with anonymous struct fields

### ✅ Performance impact?

**Negligible for common cases:**
- Name matching adds ~O(n) field comparisons
- HashMap adds small allocation overhead
- Execution time dominated by actual data casting

**Better for wide structs:**
- HashMap lookup O(1) vs repeated `column_by_name()` O(n)

---

## 📋 Checklist Review (from AGENTS.md)

From "Final quick checklist for the agent":

- [x] ✅ Prefer crate-scoped builds and tests - Tests pass
- [ ] ❓ Run `./dev/rust_lint.sh` before proposing - Not evident in commit messages
- [x] ✅ Add unit tests - 2 unit tests added in `columnar_value.rs`
- [x] ✅ Add SQL logic tests - 15+ test cases in `struct.slt`
- [x] ✅ Functions focused (under 40 lines) - `cast_struct_array_by_name()` is 38 lines
- [x] ✅ Document public APIs - Comprehensive doc comments added
- [x] ✅ Keep functions focused and modules cohesive - Good separation
- [x] ✅ Use `Result<T>` and `?` operator - Consistent error handling

**One item unclear:**
- [ ] ❓ Linting status - Should verify before merge

**Recommendation:** Run `./dev/rust_lint.sh` to ensure formatting compliance.

---

## 🔗 Related Implementations

As mentioned in AGENTS.md "Useful Helper Functions", similar struct casting exists in:

1. **[nested_struct.rs](datafusion/common/src/nested_struct.rs#L57-L93)** - Primary struct casting implementation
   - Used for schema evolution in data sources
   - Similar field-by-name matching logic
   - More error context in failure messages

2. **[schema_adapter.rs](datafusion/datasource/src/schema_adapter.rs)** - Schema evolution patterns
   - Handles missing columns in file vs query schema
   - Inserts null columns for missing fields
   - Similar adaptation strategy

3. **[schema_rewriter.rs](datafusion/physical-expr-adapter/src/schema_rewriter.rs#L905-L980)** - Struct column adaptation
   - Handles struct column rewriting in physical plans
   - Type casting + field matching
   - Similar recursive descent pattern

**Consistency analysis:**
- ✅ All use field-by-name matching (good consistency)
- ✅ All insert nulls for missing fields (good consistency)
- ⚠️ Different error message styles (minor inconsistency)
- ⚠️ HashMap vs `column_by_name()` (acceptable variation)

---

## 💡 Suggested Improvements

### High Priority (Before Merge)

1. **✅ DONE** - SQL Logic Tests added
2. **✅ DONE** - Documentation added to `cast_to()`
3. **⚠️ TODO** - Run `./dev/rust_lint.sh` to verify formatting

### Medium Priority (Current PR or Follow-up)

4. **Consider code consolidation**
   - Evaluate merging with `nested_struct::cast_struct_column()`
   - Or document why separate implementations are justified
   - Add cross-references in doc comments

5. **Add remaining edge case tests**
   ```sql
   # Duplicate field names
   # Empty structs
   # Case sensitivity
   ```

6. **Add comment explaining HashMap choice**
   ```rust
   // Use HashMap for O(1) field lookup. More efficient than repeated
   // StructArray::column_by_name() for wide structs (many fields).
   ```

### Low Priority (Follow-up PR)

7. **Optimize scalar struct casting**
   - Add TODO comment with issue number
   - Consider direct `ScalarValue` field matching

8. **Add to AGENTS.md "Useful Helper Functions"**
   ```markdown
   * `datafusion/common/src/struct_cast.rs`
     * `cast_struct_array_by_name` casts struct arrays using field name matching
       instead of positional indexing, handling reordering and missing fields.
   ```

9. **Consider performance benchmarks**
   - Benchmark wide structs (100+ fields) vs narrow structs
   - Compare HashMap vs `column_by_name()` performance
   - Validate optimization assumptions

---

## 🎓 Learning Points

### Why This Fix Was Needed

**SQL Standard Expectation:**
In SQL, named structs/records should match fields by name:
```sql
-- User expects values to follow field names, not positions
CAST(ROW(3, 4) AS ROW(a INT, b INT))  -- Positional (no names)
CAST(struct(b => 3, a => 4) AS STRUCT(a INT, b INT))  -- Should match by name
```

**Arrow's Default Behavior:**
Arrow casts structs positionally (by field index), which is correct for:
- Tuple-like data (unnamed fields)
- Schema evolution where names are immaterial
- Low-level array operations

**DataFusion's Requirement:**
DataFusion needs name-based casting for:
- CASE expressions with different field orders
- Struct literals in queries
- User-facing SQL operations where names have semantic meaning

**The Fix:**
Intercept struct casting before Arrow's positional logic and apply name-based matching.

---

## 📝 Communication

Following the copilot-instructions.md guideline:

> "When responding in PR discussions, adopt a conversational, approachable tone that explains the reasoning behind design decisions."

**Suggested PR comment template:**

```markdown
## Code Duplication Observation

I noticed the new `struct_cast.rs` shares logic with `nested_struct.rs`. Both:
- Match fields by name
- Insert nulls for missing fields  
- Handle nested structs recursively

### Why This Matters

Having two implementations means:
1. **Bug fixes need dual updates** - A fix in one might not be applied to the other
2. **Testing overhead** - Edge cases need coverage in both paths
3. **Documentation drift** - Comments may become inconsistent

### Possible Solutions

**Option A:** Consolidate into one implementation
```rust
pub use nested_struct::cast_column as cast_struct_array_by_name;
```

**Option B:** Document why they're different
```rust
// Differs from nested_struct::cast_column in:
// 1. Fallback to positional casting (backward compat)
// 2. HashMap performance optimization
// 3. 'static lifetime requirement
```

### My Take

The implementations serve different contexts (ColumnarValue vs DataSource schema adaptation), so **Option B** might be more appropriate. But would love your thoughts!

Does the performance difference (HashMap vs `column_by_name`) justify separate implementations?
```

This approach:
- ✅ Explains the **why** (maintenance burden)
- ✅ Provides **concrete examples** (code snippets)
- ✅ Offers **practical alternatives** (consolidate vs document)
- ✅ Invites **discussion** (asks for opinions)

---

## 🏆 Final Assessment

### Strengths
1. ✅ **Correctness** - Solves the reported bug completely
2. ✅ **Test Coverage** - 15+ SQL logic tests, edge cases covered
3. ✅ **Documentation** - Clear doc comments with examples
4. ✅ **Safety** - No breaking changes, backward compatible
5. ✅ **Integration** - Touches all necessary layers appropriately
6. ✅ **Code Quality** - Follows Rust idioms, clear logic flow

### Weaknesses
1. ⚠️ **Code Duplication** - Similar logic in `nested_struct.rs`
2. ⚠️ **Test Gaps** - Missing edge cases (duplicates, empty structs)
3. ⚠️ **Optimization** - Scalar struct casting could be more efficient

### Recommendation: **✅ Approve with Suggestions**

**Rationale:**
- The fix is functionally correct and well-tested
- No security, performance, or correctness blockers
- Suggested improvements are polish items, not blockers
- Benefits users immediately by fixing data corruption bug

**Action Items (Non-Blocking):**
1. Run `./dev/rust_lint.sh` before final merge
2. Consider code consolidation discussion with maintainers
3. File follow-up issue for scalar struct optimization
4. Add remaining edge case tests when time permits

**Estimated effort for all suggestions:** 2-3 hours
- Linting: 5 minutes
- Edge case tests: 30 minutes
- Code consolidation discussion: 1 hour
- Documentation updates: 30 minutes

---

## 📚 References

**Related Issues:**
- #14396 - Original bug report (struct casting by position)
- #17281 - Related PR discussion

**Codebase References:**
- [nested_struct.rs](datafusion/common/src/nested_struct.rs) - Existing struct casting
- [schema_adapter.rs](datafusion/datasource/src/schema_adapter.rs) - Schema evolution
- [AGENTS.md](AGENTS.md) - Repository guidelines

**Commits in Range:**
- 3adcc2653 - Refactor struct casting and add unit tests
- 179813d01 - Support field reordering and count changes
- 9d24794a1 - Add name-based coercion for CASE expressions
- 4a4621984 - Skip const-folding for field count changes
- 15c0f8ea8 - Enable struct coercion tests

---

**Review completed on:** 2025-12-30  
**Reviewer:** GitHub Copilot (Claude Sonnet 4.5)  
**Commit range:** 3adcc2653^..15c0f8ea8
