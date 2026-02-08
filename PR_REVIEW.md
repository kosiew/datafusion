# Code Review: CastColumnExpr Implementation with Owned Cast/Format Options

**PR**: [#20202](https://github.com/apache/datafusion/pull/20202)  
**Title**: Add schema-aware CastColumnExpr with owned cast/format options for safe struct casting  
**Review Scope**: Commits `25c2355b7^..2087e49a5`  
**Status**: ✅ **Approve with Suggestions**

---

## Executive Summary

This PR introduces a well-designed, schema-aware casting expression (`CastColumnExpr`) with owned cast/format options to support safe struct casting. The implementation is comprehensive, includes thorough validation, and maintains backward compatibility. The code is generally clean and well-documented, though a few non-blocking improvements are suggested for polish and maintainability.

---

## Strengths 🟢

### 1. **Owned Format/Cast Options - Excellent Design**

The introduction of `OwnedFormatOptions` and `OwnedCastOptions` is well-motivated and solves a real problem:

- **Problem**: Arrow's `CastOptions<'static>` requires `&'static str` references, preventing dynamic format strings
- **Solution**: Owned `String` values with conversion helpers (`as_arrow_options()`, `from_arrow_options()`)
- **Implementation**: Clean builder pattern, proper docs, derives `Default`, `Clone`, `Hash`, `Eq`, `PartialEq`

```rust
// Example: Clear API for creating dynamic options
let owned_options = OwnedCastOptions::new(true)
    .with_format_options(OwnedFormatOptions::new()
        .with_date_format(Some(user_provided_format)));
```

**Observation**: Public reexport in `datafusion/common/src/lib.rs` is correct for public API stability.

---

### 2. **Comprehensive Validation Framework**

The validation logic is broken into focused, well-documented helper functions with clear responsibility separation:

| Function | Responsibility |
|----------|-----------------|
| `validate_column_expr()` | Column index bounds + castability checks |
| `validate_expression_field()` | Expression type/nullability compatibility |
| `validate_input_to_target_cast()` | Input→target transformation legality |
| `validate_cast_compatibility()` | Orchestrates all validation steps |

Each function has **clear documentation** explaining what is validated and why. This makes the logic easy to understand and audit.

**Example error messages** are precise:
```
CastColumnExpr column 'b' at index 1 has data type 'Struct("nested": Int32)' 
which cannot be cast to input field type 'Int32'
```

---

### 3. **Correct Schema-Aware Construction**

The PR correctly fixes a critical bug in `PhysicalExprAdapter`:

**Before**:
```rust
let actual_physical_field = self.physical_file_schema.field(column.index());
```
❌ Assumes logical column index matches physical schema position

**After**:
```rust
let physical_column_index = self.physical_file_schema.index_of(column.name())?;
let actual_physical_field = self.physical_file_schema.field(physical_column_index);
```
✅ Resolves by **column name**, handling schema column reordering

---

### 4. **Dual Constructor Design**

Two factory methods serve distinct use cases well:

```rust
// Simple case: single column
pub fn new(...) -> Result<Self>
    // Creates single-field schema internally

// Complex case: column in larger schema  
pub fn new_with_schema(..., input_schema: Arc<Schema>) -> Result<Self>
    // Accepts full schema (useful for deserialization)
```

This captures the common case while supporting protobuf round-tripping cleanly.

---

### 5. **Protobuf Support - Backward Compatible**

The protobuf integration is thorough:

- ✅ New `PhysicalCastColumnNode` with dedicated `cast_column` tag (22)
- ✅ New `PhysicalCastOptions` and `FormatOptions` messages  
- ✅ Backward compatibility via **deprecated fields** (`safe`, `format_options`) with fallback logic
- ✅ `DurationFormat` enum for proper format serialization (ISO8601, Pretty)
- ✅ Clear comments explaining deprecation policy

**Fallback logic** is correct:
```rust
match cast_options {
    Some(opts) => { /* use new field */ }
    None => {
        // Fallback to deprecated fields for backward compatibility
        if format_options.is_some() || safe { /* use old */ }
        else { Ok(None) }
    }
}
```

---

### 6. **Tests Cover Happy Paths + Error Cases**

Test coverage includes:
- ✅ Primitive type casting (Int32→Int64)
- ✅ Struct with field reordering
- ✅ Nested struct casting  
- ✅ Scalar struct wrapping
- ✅ **Negative tests**: schema mismatches, nullability narrowing

```rust
#[test]
fn cast_column_schema_mismatch_nullability_metadata() {
    // Correctly rejects nullable -> non-nullable casts
    let err = CastColumnExpr::new_with_schema(...)
        .expect_err("should reject nullable -> non-nullable cast");
    assert_contains!(err.to_string(), "Cannot cast nullable struct field");
}
```

---

### 7. **Nullability Alignment Fix**

The PR systematically updates schema nullability across examples and tests:
- File schema fields are **nullable** by default (matching real data)
- Logical table schemas adjusted to avoid unsafe narrowing
- Example: `custom_file_casts.rs` updated Int32 field to nullable ✅

This prevents data loss from narrowing nullable→non-nullable conversions.

---

## Suggestions for Enhancement 📝

### 1. **Extract `normalize_cast_options` Pattern (Polish)**

**Current Code**:
```rust
fn normalize_cast_options(cast_options: Option<OwnedCastOptions>) -> OwnedCastOptions {
    cast_options.unwrap_or_default()
}
```

**Suggestion**: This single-line function is fine as-is for documentation, but consider:
- If reused across module, make it a public helper
- Or inline with a comment: `let cast_options = cast_options.unwrap_or_default();`

**Impact**: Minor code clarity improvement. Current implementation is acceptable.

---

### 2. **Schema Field Resolution Helper (Future Consideration)**

The pattern of `index_of()` + `field()` appears in multiple places. Consider adding a small helper:

**Current** (in `schema_rewriter.rs`):
```rust
let physical_column_index = self.physical_file_schema.index_of(column.name())?;
let actual_physical_field = self.physical_file_schema.field(physical_column_index);
```

**Future Helper** (not required now):
```rust
impl Schema {
    pub fn field_by_name(&self, name: &str) -> Result<&Field> {
        let index = self.index_of(name)?;
        Ok(self.field(index))
    }
}
```

**Context**: If this pattern appears frequently in future PRs, consider extracting. Currently, the explicit code is clear enough.

---

### 3. **Protobuf Duration Format Handling - Hardening (Minor)**

**Current Code** (in `from_proto.rs`):
```rust
fn parse_duration_format(format_str: Option<&str>) -> DurationFormat {
    match format_str {
        Some("iso8601") => DurationFormat::ISO8601,
        _ => DurationFormat::Pretty, // Default to Pretty
    }
}
```

**Observation**: The `_` matches both `None` and unrecognized values silently. This is acceptable (defaults to Pretty), but could log a warning for diagnostic purposes in debug builds.

**Suggestion (optional)**: 
```rust
fn parse_duration_format(format_str: Option<&str>) -> DurationFormat {
    match format_str {
        Some("iso8601") => DurationFormat::ISO8601,
        Some(unknown) => {
            debug!("Unknown duration format: {}, defaulting to Pretty", unknown);
            DurationFormat::Pretty
        }
        None => DurationFormat::Pretty,
    }
}
```

**Impact**: Improves debuggability. Current code is functionally correct.

---

### 4. **Consider Naming: `new_with_schema` vs `new_validated` (Discussion)**

**Current naming**:
```rust
pub fn new(...)                              // Simple single-field schema
pub fn new_with_schema(..., schema: ...) -> Result<Self>  // Full schema + `?`
```

**Observation**: The `?` return type difference is clear, but the name `new_with_schema` doesn't hint at the validation aspect.

**Alternative consideration**:
```rust
pub fn new(...) -> Result<Self>                      // Always validates
pub fn new_with_schema(...) -> Result<Self>          // Explicit schema  
pub fn unchecked(...)  // If validation-skipping is ever needed
```

**Verdict**: Current naming is appropriate. The docstrings clearly explain the validation behavior, and the return type `Result` signals validation.

---

### 5. **API Documentation - Struct-Centric Context (Minor)**

**Current Doc** (in `cast_column.rs`):
> "This expression is intended for schema rewriting scenarios where the planner already resolved the input column..."

**Suggestion**: Add a concrete example in the doc comment showing the typical usage:

```rust
/// # Examples
///
/// ```ignore
/// // When adapting a physical file schema to a logical table schema:
/// let cast = CastColumnExpr::new_with_schema(
///     Arc::new(Column::new("id", 0)),
///     Arc::new(Field::new("id", DataType::Int32, true)),   // file schema
///     Arc::new(Field::new("id", DataType::Int64, true)),   // table schema
///     None,
///     file_schema,
/// )?;
/// ```
```

**Impact**: Improves discoverability. The current docs are good; this is enhancement-level.

---

### 6. **Consider `Hash` Implementation Note (Documentation)**

The `CastColumnExpr` implements `Hash` including `input_schema`. This is correct but deserves a comment explaining why:

```rust
impl Hash for CastColumnExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Include input_schema in hash to catch mismatches in plan deduplication
        self.expr.hash(state);
        self.input_field.hash(state);
        self.target_field.hash(state);
        self.cast_options.hash(state);
        self.input_schema.hash(state);  // <-- Explains why schema is included
    }
}
```

**Impact**: Clarity for future maintainers. Current implementation is correct.

---

## Testing Observations ✅

| Aspect | Status |
|--------|--------|
| Unit tests (happy path) | ✅ Comprehensive |
| Error cases | ✅ Nullability, schema mismatch covered |
| Integration tests | ✅ Used in parquet adapter tests |
| Protobuf round-trip | ✅ Serialization + deserialization |
| Backward compatibility | ✅ Deprecated field fallback tested |

**Note**: SQL logic tests are appropriate scope for integration; no new SLT files needed here since this is infrastructure, not query functionality.

---

## Consistency Checks ✅

| Aspect | Status |
|--------|--------|
| **Style** | ✅ Rust conventions, proper use of `Arc`, `FieldRef` |
| **Naming** | ✅ Clear, descriptive (validate_*, cast_*, normalize_*) |
| **Error Handling** | ✅ Uses `plan_err!` and `Result<T>` consistently |
| **Documentation** | ✅ Doc comments on public items and validation functions |
| **Module Organization** | ✅ New types in `format.rs`, expressions in `physical-expr/` |
| **Trait Implementations** | ✅ `PhysicalExpr`, `Display`, `PartialEq`, `Hash` all present |

---

## Scope Assessment ✅

**Scope**: Focused and appropriate
- ✅ No unrelated refactoring
- ✅ Changes centered on CastColumnExpr and integration points
- ✅ Backward compatibility maintained in protobuf
- ✅ Schema nullability fixes necessary (not scope creep)

**Files Modified**: 
- Core: `datafusion/common/format.rs`, `datafusion/physical-expr/cast_column.rs`
- Integration: `datafusion/physical-expr-adapter/schema_rewriter.rs`
- Protobuf: `datafusion/proto/` (serialization/deserialization)
- Examples/Tests: Updated for nullability alignment

---

## Design Patterns Evaluation ✅

| Pattern | Evaluation |
|---------|-----------|
| **Factory Methods** | ✅ Two constructors for different use cases (appropriate) |
| **Validation-in-Constructor** | ✅ Prevents invalid states from being constructed |
| **Owned vs Borrowed Options** | ✅ Solves the `&'static str` lifetime constraint clearly |
| **Builder Pattern (OwnedFormatOptions)** | ✅ Standard Rust idiom, easy to extend |
| **Backward Compat (Protobuf)** | ✅ Proper deprecation with fallback logic |

---

## Risk Assessment

### Low Risk ✅
- CastColumnExpr is new infrastructure, not modifying existing expressions
- Validation is additive; prevents unsafe casts
- Protobuf backward compat is well-implemented

### When Merging
- Schema nullability changes in examples/tests are safe (improving alignment)
- Physical adapter fix (column name lookup) is a correctness improvement
- No breaking changes to public APIs

---

## Final Recommendations

### ✅ Ready to Approve
This PR is **ready for merge** with the following observations:

1. **Implement**: 
   - All functionality is complete and correct ✅
   - Validation is thorough ✅
   - Tests cover happy paths and error cases ✅
   - Protobuf support is backward compatible ✅

2. **Consider for Future Work** (not blocking):
   - Add inline documentation comment in `Hash` impl explaining scope inclusion
   - If `index_of()` + `field()` pattern becomes frequent, extract to `field_by_name()` helper
   - Optional: Add `debug!` logging for unrecognized duration formats in protobuf parsing

3. **After Merge**:
   - Monitor for any schema-related issues in datasource adapters (verify nullability alignment propagates correctly)
   - Consider documenting CastColumnExpr usage in library-user-guide

---

## Summary Table

| Category | Score | Notes |
|----------|-------|-------|
| **Correctness** | ✅ Excellent | Validation is comprehensive, protobuf round-trip works |
| **Code Quality** | ✅ Very Good | Clear structure, proper error messages, good docs |
| **Test Coverage** | ✅ Good | Happy paths + error cases; integration coverage via parquet tests |
| **API Design** | ✅ Good | Dual constructors well-motivated; owned options solve real problem |
| **Documentation** | ✅ Good | Public items documented; suggest adding example to struct doc |
| **Backward Compat** | ✅ Excellent | Protobuf fallback is clean and correct |
| **Scope** | ✅ Focused | No creep; related fixes (nullability) are justified |

---

## Conclusion

This is a **well-executed PR** that enhances DataFusion's casting capabilities with proper validation and schema awareness. The owned format/cast options are a clean solution to the `&'static str` lifetime constraint, and the validation framework is thorough and maintainable.

**Recommendation**: ✅ **Approve and Merge**

The suggested enhancements are polish-level improvements for future work, not blockers. The implementation is production-ready.
