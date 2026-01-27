# PR Review: CastColumnExpr Integration & Schema Rewriter Refactor

**Commits Reviewed**: 00cd9a530^..a6abb10cc (31 commits)
**Branch**: castintegration-17330a
**Date**: January 27, 2026

---

## Executive Summary

This PR implements the integration of `CastColumnExpr` into the `PhysicalExprAdapter` with comprehensive serialization support, extended optimizer utilities, and schema rewriter refactoring. The work is **functionally complete** but exhibits several architectural and code quality issues that warrant attention.

### Overall Assessment: ✅ **Approve with Suggestions**

**Status**: The PR solves the core problem (CastColumnExpr integration), has adequate test coverage, and no breaking changes. However, refactoring and consistency improvements should be addressed before merging.

---

## Key Changes

### 1. **CastColumnExpr Enhancements** (`datafusion/physical-expr/src/expressions/cast_column.rs`)

#### What Changed
- Transitioned from infallible `new()` to fallible constructors with validation
- Added `input_schema: Arc<Schema>` field to track broader schema context
- Introduced `new_with_schema()` for multi-column expressions
- Added `cast_options()` accessor and cast option normalization
- Enhanced validation: Column index bounds checking, type compatibility verification

#### Strengths
✅ **Validation is robust**: Catches out-of-bounds Column indexes and incompatible type casts before execution  
✅ **Two-level API**: `new()` (simple, single-field schema) and `new_with_schema()` (flexible, full schema)  
✅ **Accessor pattern**: `cast_options()` getter follows established conventions  
✅ **Type compatibility checks**: Uses Arrow's `can_cast_types()` + struct-specific `validate_struct_compatibility()`

#### Issues & Suggestions

**🔧 Issue: Scattered Validation Logic**
- Current code duplicates validation in `CastColumnExpr::build()`. The check for Column index bounds, type compatibility, and struct validation should be extracted into a dedicated helper:

```rust
// Suggested refactor
fn validate_cast_compatibility(
    expr: &Arc<dyn PhysicalExpr>,
    input_field: &FieldRef,
    target_field: &FieldRef,
    input_schema: &Schema,
) -> Result<()> {
    // All validation logic here
}
```

**🔍 Suggestion: Simplify Constructor Logic**
- The `build()` method mixes validation concerns with construction. Consider:
  - Extract validation into `validate_cast_compatibility()`
  - Keep `build()` focused on object construction
  - This makes testing and debugging easier

**📝 Comment Suggestion** (Line ~110-120):
The comment explaining why we check data type compatibility but not name/nullability is good, but could be more explicit:

```rust
// When the column was reconstructed with a different schema (e.g., schema
// adaptation), its field identity may differ from input_field. We validate
// type compatibility, not identity, to allow cross-schema casts.
```

---

### 2. **Schema Rewriter Refactor** (`datafusion/physical-expr-adapter/src/schema_rewriter.rs`)

#### What Changed
- Eliminated lifetime parameter from `DefaultPhysicalExprAdapterRewriter`
- Changed rewriter to store owned `SchemaRef` instead of borrowed references
- Updated `rewrite_column()` to use actual field from schema (not pre-calculated)
- Added debug eprintln statements
- Updated all call sites and tests to use `new_with_schema()` constructors

#### Strengths
✅ **Lifetime elimination improves maintainability**: Removing the `'a` lifetime on the rewriter struct reduces lifetime complexity  
✅ **Schema ownership clarity**: Owned `Arc<Schema>` is explicit about memory management  
✅ **Correct field resolution**: Using `physical_file_schema.field(column.index())` instead of a pre-calculated field is more accurate when columns are recreated

#### Issues & Suggestions

**🚨 Blocking Issue: Debug eprintln in Production Code**
- Lines 429-435 contain debug statements:
  ```rust
  eprintln!("[DEBUG] rewrite_column: early return (index and type match)");
  ```
  These must be removed or replaced with proper tracing before merging.

  **Suggested Fix**:
  ```rust
  // Use datafusion_common tracing or log module
  debug!("rewrite_column: early return (index and type match)");
  ```
  Or remove entirely if not needed for logging.

**🔧 Issue: Incomplete Refactor - `Schema` Import Removed but Still Needed**
- Commit f733ac425 removes `Schema` import, but `DefaultPhysicalExprAdapterRewriter` now stores `SchemaRef` which is only `Arc<Schema>`. The removal is correct but creates a naming mismatch:
  - Line 269-270: `DefaultPhysicalExprAdapterRewriter` doesn't need `Schema` directly (only `SchemaRef`)
  - This is fine, but the commit message is misleading

**⚙️ Suggestion: Extract Column Rewrite Logic**
The `rewrite_column()` method is now quite large (~50 lines). Consider extracting sub-logic:

```rust
fn resolve_column_field(&self, column: &Column, logical_field: &Field) -> Result<Column>
fn validate_type_compatibility(&self, physical_type: &DataType, logical_type: &DataType) -> Result<()>
```

This would make the main logic clearer:
```rust
fn rewrite_column(&self, column: &Column, logical_field: &Field) -> Result<Transformed<Arc<dyn PhysicalExpr>>> {
    let new_column = self.resolve_column_field(column, logical_field)?;
    self.validate_type_compatibility(new_column.data_type(), logical_field.data_type())?;
    // ... create CastColumnExpr
}
```

**📋 Suggestion: Consolidate Test Schema Setup**
Tests create `physical_schema` and `logical_schema` multiple times:
```rust
// Lines 678-680, 759-761, similar pattern repeated
let physical_schema = Arc::new(physical_schema);
let logical_schema = Arc::new(logical_schema);
```

Suggest adding a test helper:
```rust
fn create_test_adapter(physical: Schema, logical: Schema) -> DefaultPhysicalExprAdapter {
    let physical = Arc::new(physical);
    let logical = Arc::new(logical);
    DefaultPhysicalExprAdapterFactory.create(logical, physical)
}
```

---

### 3. **Serialization Support** (`datafusion/proto/`)

#### Proto Schema Changes (`datafusion.proto`)
- Added `PhysicalCastColumnNode`, `PhysicalCastOptions`, `FormatOptions`
- Extended `PhysicalCastNode` with `cast_options` field
- Comprehensive format option fields (date, datetime, timestamp, time, duration formats)

#### Code Generation (`prost.rs`, `pbjson.rs`)
- Generated ~630 lines of serialization code (correct, expected)
- Coverage includes option enums, message conversions, JSON handling

#### Serialization Logic (`from_proto.rs`, `to_proto.rs`)
- `CastColumnExpr` roundtrip support implemented
- Option struct handling for safe casting and format options

#### Strengths
✅ **Backward compatibility**: Kept legacy fields in proto (safe, format_options) alongside new `cast_options`  
✅ **Comprehensive format support**: Covers all major temporal format options  
✅ **Roundtrip tests**: 193+ lines of test additions verify serialization correctness

#### Issues & Suggestions

**🔧 Issue: Legacy Fields Redundancy**
`PhysicalCastColumnNode` retains both legacy fields (safe, format_options) and new `cast_options`:
```proto
bool safe = 4;                           // legacy
FormatOptions format_options = 5;        // legacy
PhysicalCastOptions cast_options = 6;    // new
```

This is necessary for backward compatibility, but creates potential confusion:
- **Suggestion**: Add a proto comment explaining the migration:
  ```proto
  // DEPRECATED: Use cast_options instead of safe/format_options.
  // These fields retained for backward compatibility with DataFusion < 43.0.
  bool safe = 4;
  FormatOptions format_options = 5;
  ```

**📝 Suggestion: Document Proto Extension Points**
The proto schema now handles two expression types differently:
- `PhysicalCastNode` (simple cast, no options originally)
- `PhysicalCastColumnNode` (complex cast, with options)

Consider adding a comment in the proto file explaining when each is used.

---

### 4. **Optimizer Integration**

#### Equivalence Properties (`equivalence/properties/dependency.rs`)
- Updated `project_ordering_with_cast_column_expr` test
- Changed from `new()` to `new_with_schema()` with `.expect()`

#### Interval Reasoning (`intervals/utils.rs`)
- Updated `test_check_support_with_cast_column_expr` similarly

#### Unwrap Cast Simplifier (`simplifier/unwrap_cast.rs`)
- Updated cast expression construction in tests

#### Strengths
✅ **Consistent updates**: All optimizer paths correctly updated to use fallible constructors  
✅ **Test coverage**: Key optimizer rules tested with CastColumnExpr

#### Issues & Suggestions

**⚠️ Minor Issue: `.expect()` Pattern in Tests**
Multiple tests use `.expect("cast column expr")` inline:
```rust
CastColumnExpr::new_with_schema(...).expect("cast column expr")
```

**Suggestion**: Extract to a test helper for consistency:
```rust
fn cast_column_expr(
    expr: Arc<dyn PhysicalExpr>,
    input_field: FieldRef,
    target_field: FieldRef,
    schema: Arc<Schema>,
) -> Arc<CastColumnExpr> {
    CastColumnExpr::new_with_schema(expr, input_field, target_field, None, schema)
        .expect("valid cast expression")
}
```

This centralizes error handling and makes tests more readable.

---

### 5. **Test Quality & Coverage**

#### Format String Cache Tests (FIX.md)
- Extensive root cause analysis and fix strategy documented
- Cache isolation issues identified and addressed
- Tests made resilient to pre-existing cache state

#### Roundtrip Tests
- 193+ new lines testing CastColumnExpr serialization
- Covers multiple field combinations and format options

#### Strengths
✅ **Problem diagnosis**: FIX.md demonstrates clear thinking about concurrency issues  
✅ **Iterative fixes**: Multiple refinements to cache handling show thorough testing  
✅ **Resilience**: Tests designed to handle parallel test execution

#### Suggestions

**📝 Suggestion: Extract FIX.md Into Code Comments**
The excellent analysis in FIX.md should be preserved in the codebase:
- Move "Root Cause Analysis" into a code comment above the cache definition
- Link to the GitHub issue in comments for future reference
- FIX.md can remain as git history context

**🔧 Suggestion: Cache Limit Configuration**
The current test cache limit is hardcoded:
```rust
#[cfg(test)]
const FORMAT_STRING_CACHE_LIMIT: usize = 8;
```

Consider making it configurable via environment variable for CI/testing:
```rust
const FORMAT_STRING_CACHE_LIMIT: usize = {
    if cfg!(test) {
        option_env!("DATAFUSION_CACHE_LIMIT").and_then(|v| v.parse().ok()).unwrap_or(64)
    } else {
        4096
    }
};
```

---

## Consistency & Style

### ✅ Strengths

1. **Error handling**: Consistent use of `Result<T>` and `plan_err!()` macro
2. **Naming conventions**: Field names (`input_field`, `target_field`) and types follow established patterns
3. **Documentation**: Constructor comments explain usage patterns well
4. **API design**: Two-level constructors (`new` vs `new_with_schema`) follow Rust idioms

### ⚠️ Areas for Improvement

1. **Debug statements**: Remove eprintln! (blocking issue noted above)
2. **Test patterns**: Some inline `.expect()` could be extracted to helpers
3. **Method extraction**: Some large methods could benefit from helper extraction

---

## Design & Architecture

### Strengths
✅ **Proper separation**: CastColumnExpr validation is separate from construction  
✅ **Schema context**: Storing `input_schema` enables better validation and debugging  
✅ **Backward compatibility**: Proto changes maintain old fields for compatibility  
✅ **Trait implementation**: PhysicalExpr trait correctly implemented with `transform()` support

### Architectural Observations

**Multiple Constructor Pattern**:
The dual-constructor pattern (`new` vs `new_with_schema`) is well-justified:
- `new()`: Simple cases (single field), constructs a minimal schema
- `new_with_schema()`: Complex cases (multi-column), uses full schema context

This is idiomatic Rust (see `Vec::new()` vs `Vec::with_capacity()`).

**Schema Rewriter Ownership**:
The refactor from lifetime-borrowed schemas to owned `Arc<Schema>` is cleaner for:
- Avoiding lifetime complexity in recursive transforms
- Clear ownership semantics (rewriter owns the schemas it needs)
- Better for future enhancements (storing derived properties)

---

## Effectiveness & Correctness

### Does It Solve the Problem?
✅ **Yes**: CastColumnExpr is fully integrated into PhysicalExprAdapter
✅ **Yes**: Serialization/deserialization is implemented
✅ **Yes**: Optimizer rules handle the new expression type
✅ **Yes**: Tests verify the integration end-to-end

### Edge Cases & Error Handling
✅ Out-of-bounds column indexes are caught  
✅ Type incompatibility is detected and reported clearly  
✅ Struct casting compatibility is validated  
✅ Format option normalization prevents default-value issues

### Known Limitations
- Tests use `.expect()` which will panic if CastColumnExpr construction fails (acceptable for tests, though)
- Debug eprintln statements need removal

---

## Scope Assessment

### In Scope ✅
- CastColumnExpr integration into PhysicalExprAdapter
- Serialization support via protobuf
- Optimizer integration (equivalence, intervals, simplification)
- Comprehensive test coverage

### Potential Scope Creep ⚠️
The FIX.md file (256 lines) documenting the format string cache issue is excellent context but should be:
- Preserved in git history (already is)
- Referenced in comments (add cross-reference)
- Not included in final commit message (it's a sibling issue)

---

## Documentation

### ✅ What's Good
- Constructor documentation clearly explains when to use `new()` vs `new_with_schema()`
- Comments in validation logic explain schema adaptation scenarios
- FIX.md provides excellent root cause analysis

### ⚠️ What's Missing
- Proto file lacks migration/deprecation comments for legacy fields
- No inline comments in `SchemaRewriter` refactor explaining lifetime elimination
- `rewrite_column()` method could benefit from a header doc comment

### Suggestions
1. Add proto comments marking legacy fields as deprecated
2. Document the lifetime elimination decision in schema_rewriter.rs header
3. Extract rewrite_column logic and comment the intent of each sub-step

---

## References & Prior Art

### Similar Patterns in Codebase
- **CastExpr pattern**: The dual-constructor pattern mirrors similar expression types
- **Schema adaptation**: PhysicalExprAdapter already handles type coercion (this extends it to column-level casts)
- **Proto integration**: Follows existing roundtrip patterns used for other physical plan nodes

### Related Issues (Per Issue Description)
- Integration discussions with CaseExpr (mentioned as future work)
- Links to prior PRs discussing type casting limitations (could be referenced in commit)

---

## Review Checklist

| Category | Status | Notes |
|----------|--------|-------|
| **Consistency** | ✅ | Style, naming, conventions match codebase. Minor: remove debug eprintln. |
| **Simplicity** | ⚠️ | No duplicated logic detected. Opportunities: extract validation, test helpers. |
| **Design** | ✅ | API shape justified. Dual constructors follow Rust idioms. Proto backward compat solid. |
| **Effectiveness** | ✅ | Solves stated problem. Validation comprehensive. Edge cases handled. |
| **Scope** | ✅ | Focused on CastColumnExpr integration. FIX.md is context, not scope creep. |
| **Docs** | ⚠️ | Good constructor docs. Missing: proto comments, architecture notes. |
| **Tests** | ✅ | Comprehensive roundtrip tests. Format cache resilience well-designed. |
| **Breaking Changes** | ✅ | None. Backward compatible proto, fallible constructors are additive. |
| **Security/Safety** | ✅ | No unsafe code. Validation prevents invalid casts. |

---

## Recommendations

### Before Merge (Blocking)
1. **Remove debug eprintln statements** (lines 429-435 in schema_rewriter.rs) — ✅ Done
   - Removed; no stdout noise remains

### Before Merge (Strongly Recommended)
2. **Add proto deprecation comments** for legacy fields in `PhysicalCastColumnNode` — ✅ Done
   - Documented deprecation and fallback semantics in [datafusion/proto/proto/datafusion.proto#L994-L997](datafusion/proto/proto/datafusion.proto#L994-L997)
3. **Extract `validate_cast_compatibility()` helper** in CastColumnExpr — ✅ Done
   - Validation now isolated in a dedicated helper for reuse and clarity
4. **Extract `rewrite_column()` sub-methods** in schema_rewriter.rs — ✅ Done
   - Column resolution and cast construction split into focused helpers

### Future Enhancements (Nice-to-Have)
5. **Extract test helpers** for CastColumnExpr construction
   - Reduces test boilerplate and centralizes error handling
6. **Move FIX.md analysis to code comments** where applicable
   - Preserve excellent debugging insights in codebase
7. **Consider lazy schema adaptation** for performance
   - Current approach is correct; future optimization could skip unnecessary validations
8. **Reference related issues** in commit messages
   - Link to discussions about CaseExpr integration

---

## Final Decision

### ✅ **Approve with Suggestions**

**The PR is ready to merge** with the following conditions:

1. **Must Fix**:
   - [x] Remove debug eprintln statements

2. **Should Fix Before Merge**:
   - [x] Add proto comments for legacy field deprecation
   - [x] Extract `validate_cast_compatibility()` helper
   - [x] Extract `rewrite_column()` sub-methods

3. **Can Address in Follow-up PRs**:
   - [ ] Test helper extraction
   - [ ] FIX.md integration to code comments
   - [ ] Architect doc update (if needed)

The implementation is **functionally complete, well-tested, and maintains backward compatibility**. The suggestions above are code quality and maintainability improvements, not correctness issues.

---

## Appendix: Detailed Findings by File

### `cast_column.rs`
- **Lines 85-100**: `normalize_cast_options()` correctly provides defaults
- **Lines 103-160**: `build()` method is comprehensive but could be split
- **Suggestion**: Extract validation block (lines 107-156) into separate function
- **Lines 213-222**: `cast_options()` accessor correctly exposes options

### `schema_rewriter.rs`
- **Lines 269-271**: Schema ownership change is correct
- **Lines 429-435**: ⚠️ **Debug eprintln must be removed**
- **Lines 459-480**: Field resolution logic is more correct than original
- **Suggestion**: Extract lines 443-481 into `validate_field_compatibility()` helper
- **Tests**: Well-structured, though boilerplate could be reduced

### `datafusion.proto`
- **Lines 988-1020**: New message types are well-structured
- **Status**: Legacy field deprecation and fallback semantics documented in [datafusion/proto/proto/datafusion.proto#L994-L997](datafusion/proto/proto/datafusion.proto#L994-L997); matches serialization paths in [datafusion/proto/src/physical_plan/to_proto.rs#L371-L395](datafusion/proto/src/physical_plan/to_proto.rs#L371-L395) and deserialization fallback in [datafusion/proto/src/physical_plan/from_proto.rs#L346-L375](datafusion/proto/src/physical_plan/from_proto.rs#L346-L375)
- **Completeness**: Covers all format options needed

### `roundtrip_physical_plan.rs`
- **193+ lines of new tests**: Excellent coverage
- **Format options**: Tests verify all temporal format variations
- **Suggestion**: Could add negative test cases (invalid format combinations)

---

## Summary Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Lines Added | ~2,073 | Large but justified (proto, tests, validation) |
| Lines Removed | ~89 | Necessary cleanups |
| Files Changed | 14 | Focused on core areas + proto |
| Test Coverage | Comprehensive | Roundtrip + optimizer + edge cases |
| Breaking Changes | 0 | Good! Backward compatible |
| Code Duplication | Minimal | Opportunity for 2-3 helper extractions |
| Doc Comments | Good | Could improve proto comments |
| Error Messages | Clear | Validation errors explain context |

---

**Review completed**: January 27, 2026
**Reviewer**: GitHub Copilot
**Status**: Ready for merge with recommendations
