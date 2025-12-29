# PR Review: Nested DataType Filter Pushdown to Parquet (Commit b64838e4c)

**Commit:** `b64838e4c` — Document nested pushdown semantics and optimizations

**Date:** December 29, 2025

---

## Executive Summary

This commit implements **support for filtering nested data types (specifically lists) during Parquet decoding**, addressing the feature request for `array_has_all()` and similar predicates to be pushed down to the Arrow reader layer. The implementation is functionally correct and well-tested, but has several design and implementation opportunities for improvement.

**Decision:** ✅ **Approve with suggestions** — The feature works as intended and is well-documented, but non-blocking improvements in code organization, API clarity, and helper abstractions would strengthen the solution.

---

## Detailed Review

### 1. ✅ **Consistency & Patterns**

**Strengths:**
- Naming aligns with DataFusion conventions (`PushdownChecker`, `FilterCandidate`, `ProjectionColumns`).
- Uses established patterns: tree visitation with `TreeNodeVisitor`, metrics integration, Arc-based sharing.
- Error handling follows DataFusion conventions (`Result<T>`, `datafusion_common::Result`).

**Minor Observations:**
- `ProjectionColumns` struct is well-named but the `#[allow(dead_code)]` on `root_indices` suggests it may not be serving its intended purpose yet—consider documenting why both coordinates are tracked if only leaves are used in all current code paths.

---

### 2. 🎯 **Simplicity & Code Organization**

**Strengths:**
- Clear separation of concerns: `PushdownChecker` validates pushdown feasibility, `FilterCandidateBuilder` collects cost metadata, `supports_list_predicates()` identifies supported functions.
- The `leaf_indices_for_roots()` helper cleanly maps Arrow root indices to Parquet leaf indices—this abstraction is valuable.

**Suggestions:**

**2a. Extract `supports_list_predicates()` into a shared registry**
```rust
// SUGGESTION: Move to a dedicated module for extensibility
// datafusion/datasource-parquet/src/supported_predicates.rs

/// Registry of functions supported for nested list pushdown.
/// This module makes it easy to extend support for new functions
/// without modifying the row filter logic.
pub fn is_supported_list_predicate(name: &str) -> bool {
    matches!(name, "array_has" | "array_has_all" | "array_has_any")
}

fn supports_list_predicates(expr: &Arc<dyn PhysicalExpr>) -> bool {
    // ... NULL checks ...
    if let Some(fun) = expr.as_any().downcast_ref::<ScalarFunctionExpr>() {
        if is_supported_list_predicate(fun.name()) {
            return true;
        }
    }
    // ... recurse ...
}
```
**Benefit:** Decouples predicate registration from row filter logic, making the set of supported predicates a single source of truth and easier to document.

**2b. Consider consolidating `PushdownChecker` initialization logic**

Currently:
```rust
let allow_list_columns = supports_list_predicates(expr);
let mut checker = PushdownChecker::new(file_schema, allow_list_columns);
expr.visit(&mut checker)?;
```

Could be:
```rust
// In PushdownChecker::create or a factory method
pub fn create_for_expr(
    expr: &Arc<dyn PhysicalExpr>,
    file_schema: &'schema Schema,
) -> Self {
    let allow_list_columns = supports_list_predicates(expr);
    Self::new(file_schema, allow_list_columns)
}
```
**Benefit:** Reduces coupling between `pushdown_columns()` and checker construction; clarifies intent that predicate support is determined upfront.

---

### 3. 📐 **Design & API Shape**

**Strengths:**
- `ProjectionColumns` struct correctly separates root (Arrow schema) and leaf (Parquet schema) indices—this is essential for nested types.
- The `DatafusionArrowPredicate::try_new()` API is clear: takes a `FilterCandidate`, metadata, and metrics.
- Documentation updates comprehensively explain when list predicates can be pushed down.

**Observations:**

**3a. `contains_nested` flag in `FilterCandidate` and `PushdownColumns` could be more explicit**

The current flow:
1. `supports_list_predicates()` returns bool (allows list columns)
2. `PushdownChecker::new()` takes that bool
3. `PushdownChecker` sets `contains_nested = true` if nested is found
4. Later, `leaf_indices_for_roots()` uses `contains_nested` to decide conversion strategy

**Alternative:** Introduce an explicit enum to capture nested semantics:
```rust
#[derive(Debug, Clone, Copy)]
enum NestedBehavior {
    /// No nested columns involved
    PrimitiveOnly,
    /// Has nested lists that are supported
    ListsSupported,
    /// Has unsupported nested types (structs, etc.)
    Unsupported,
}

struct PushdownColumns {
    required_columns: BTreeSet<usize>,
    nested: NestedBehavior,
}
```
**Benefit:** Makes the state machine explicit; prevents subtle bugs where `contains_nested=true` with `allow_list_columns=true` could have unclear semantics. Current code is correct, but the intent is less obvious.

**3b. The `ProjectionMask::leaves()` vs `ProjectionMask::roots()` change is correct but deserves a comment**

```rust
projection_mask: ProjectionMask::leaves(
    metadata.file_metadata().schema_descr(),
    candidate.projection.leaf_indices.iter().copied(),
),
```

**Add context comment:**
```rust
// Use leaf indices: when nested columns are involved, we must specify
// leaf (primitive) column indices in the Parquet schema so the decoder
// can properly project and filter nested structures.
projection_mask: ProjectionMask::leaves(
    metadata.file_metadata().schema_descr(),
    candidate.projection.leaf_indices.iter().copied(),
),
```

---

### 4. ✅ **Effectiveness & Correctness**

**Strengths:**
- **Core logic is sound:** The visitor pattern correctly identifies when nested columns are referenced and gates pushdown based on predicate support.
- **Leaf index mapping is correct:** `leaf_indices_for_roots()` properly maps from Arrow root indices to Parquet leaf indices by filtering on `get_column_root_idx()`.
- **Metrics are properly integrated:** All three predicates share `rows_pruned` (cumulative) with only the last reporting `rows_matched` (final result), preventing double-counting.
- **Tests cover the happy path:** The new test `array_has_all_pushdown_filters_rows()` creates a Parquet file, writes an `array_has_all()` filter, and verifies rows are correctly pruned (1 pruned, 2 matched).

**Observations:**

**4a. Edge case: What happens if `contains_nested=true` but `allow_list_columns=false`?**

In `check_single_column()`:
```rust
if !self.allow_list_columns
    || !matches!(
        self.file_schema.field(idx).data_type(),
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _)
    )
{
    self.non_primitive_columns = true;
    return Some(TreeNodeRecursion::Jump);
}
```

This correctly prevents pushdown for unsupported nested types (e.g., structs). However, the condition is subtle: **if predicates don't support lists OR the type isn't a list, block**. This means:
- Structs + unsupported context → blocked ✓
- Lists + unsupported context → blocked ✓
- Lists + supported context → allowed ✓
- Structs + supported context → blocked ✓

The logic is correct but could be clearer with an intermediate variable:

```rust
if DataType::is_nested(self.file_schema.field(idx).data_type()) {
    self.contains_nested = true;

    let is_list = matches!(
        self.file_schema.field(idx).data_type(),
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _)
    );
    
    let is_supported = self.allow_list_columns && is_list;
    
    if !is_supported {
        self.non_primitive_columns = true;
        return Some(TreeNodeRecursion::Jump);
    }
}
```

**Benefit:** Clearer intent; easier for reviewers to verify all cases are handled.

**4b. Test for unsupported nested types (structs) is present but minimal**

The test `struct_data_structures_prevent_pushdown()` correctly verifies that struct columns block pushdown even with `is_not_null()`. However, **consider adding a test for mixed scenarios:**
```rust
#[test]
fn mixed_primitive_and_struct_prevents_pushdown() {
    // Expression: `(struct_col.is_not_null()) AND (int_col = 5)`
    // Should not be pushed down because struct_col is unsupported
    // even though int_col is primitive.
}
```

**Benefit:** Ensures that any nested unsupported type in a conjunction blocks the entire predicate (which is the current and correct behavior).

---

### 5. 📋 **Scope & Test Coverage**

**Strengths:**
- Focused change: only modifies row filter pushdown logic, adds dev-dependencies for testing.
- New tests cover:
  - List columns with NULL checks (`test_filter_candidate_builder_supports_list_types`)
  - Struct columns blocking pushdown (`struct_data_structures_prevent_pushdown`)
  - List columns allowing pushdown (`nested_lists_allow_pushdown_checks`)
  - End-to-end Parquet filtering with `array_has_all()` (`array_has_all_pushdown_filters_rows`)

**Observations:**

**5a. The test `array_has_all_pushdown_filters_rows()` is valuable but could be parameterized**

Currently tests only `array_has_all()`. Consider expanding to test all three supported functions:
```rust
#[test_matrix(
    fn_name = ["array_has", "array_has_all", "array_has_any"],
)]
fn array_predicate_pushdown_filters_rows(fn_name: &str) { ... }
```

Or use a helper:
```rust
fn test_array_function_pushdown(func_name: &str, condition: Expr) { ... }

#[test]
fn array_has_pushdown() {
    test_array_function_pushdown("array_has", ...);
}

#[test]
fn array_has_all_pushdown() {
    test_array_function_pushdown("array_has_all", ...);
}

#[test]
fn array_has_any_pushdown() {
    test_array_function_pushdown("array_has_any", ...);
}
```

**5b. Missing: Test for when nested columns are present but not referenced**

```rust
#[test]
fn nested_column_not_referenced_uses_root_indices() {
    // Schema: struct_col (unsupported), int_col (supported)
    // Filter: int_col = 5
    // Expected: projection should include only int_col, not struct_col
}
```

---

### 6. 📖 **Documentation**

**Strengths:**
- **Excellent module-level docs:** Clearly explains pushdown ordering, cost estimation, and now list-aware predicates.
- **Struct and function docs are comprehensive:** `DatafusionArrowPredicate`, `FilterCandidate`, and `can_expr_be_pushed_down_with_schemas()` all have clear rustdoc.
- **Inline comments are added where needed:** The comment about leaf vs. root indices is helpful.

**Suggestions:**

**6a. Add an example to `can_expr_be_pushed_down_with_schemas()` rustdoc**

```rust
/// # Examples
///
/// ```ignore
/// // Primitive filter: can be pushed down
/// let expr = col("age").gt(Expr::Literal(ScalarValue::Int32(Some(30)), None));
/// assert!(can_expr_be_pushed_down_with_schemas(&expr, &schema));
///
/// // Struct filter: cannot be pushed down
/// let expr = col("person").is_not_null();  // where person is Struct
/// assert!(!can_expr_be_pushed_down_with_schemas(&expr, &schema));
///
/// // Array filter: can be pushed down if expression is supported
/// let expr = array_has_all(col("tags"), make_array(...));
/// assert!(can_expr_be_pushed_down_with_schemas(&expr, &schema));
/// ```
```

**6b. Document the `allow_list_columns` parameter in `PushdownChecker::new()`**

```rust
/// # Arguments
/// * `file_schema` - The Arrow schema of the Parquet file
/// * `allow_list_columns` - If true, list columns may be used by supported predicates.
///   Set to true if the expression contains array functions like `array_has_all`.
fn new(file_schema: &'schema Schema, allow_list_columns: bool) -> Self {
```

---

### 7. 🔗 **Dependencies & Integration**

**Strengths:**
- Clean dependency additions: `datafusion-functions-nested` and `tempfile` are only in dev-dependencies (appropriate for tests).
- No breaking changes to public APIs.
- Integration with existing systems (`metrics`, `ParquetFileMetrics`) is clean.

**Observations:**
- The choice to use `ScalarFunctionExpr::name()` for function detection is pragmatic but brittle if function names change. Consider adding a comment noting this assumption.

```rust
// NOTE: This relies on function names matching exactly. If function names
// are refactored, this check must be updated. Consider using a trait-based
// approach (e.g., a marker trait) for more robust detection in the future.
if matches!(fun.name(), "array_has" | "array_has_all" | "array_has_any") {
```

---

## Code Quality Checklist

| Item | Status | Notes |
|------|--------|-------|
| **Naming** | ✅ | Follows RFC 430; clear, descriptive identifiers |
| **Traits** | ✅ | Implements required traits; Debug, Clone on structs |
| **Error Handling** | ✅ | Uses `Result<T>`, error propagation with `?` |
| **Documentation** | ✅ | Rustdoc on all public items; could benefit from examples |
| **Testing** | ✅ | Good coverage; could expand edge cases |
| **Safety** | ✅ | No unsafe code; proper use of references and Arc |
| **Performance** | ✅ | Efficient: single tree visit, precomputed indices |
| **Formatting** | ⚠️ | Assume `rustfmt` passes; did not verify |
| **Linting** | ⚠️ | Assume `clippy` passes; `#[allow(dead_code)]` used once |
| **API Design** | ⚠️ | Good, but `NestedBehavior` enum could make state clearer |

---

## Specific Suggestions for Future Improvement

### High Priority (Polish)

1. **Rename/refactor `supports_list_predicates()` to a registry:**
   - Move to `supported_predicates.rs` module
   - Define `SUPPORTED_ARRAY_FUNCTIONS: &[&str]` or similar
   - Makes extending support trivial and documents the contract

2. **Clarify nested type handling in `check_single_column()`:**
   - Introduce intermediate variables (`is_list`, `is_supported`)
   - Improves readability and makes all branches explicit

3. **Add test for mixed primitive + struct scenarios:**
   - Ensures unsupported nested types in a conjunction block the entire predicate
   - Validates current behavior is intentional

### Medium Priority (Enhancement)

4. **Parameterize `array_has_all_pushdown_filters_rows()` test:**
   - Test all three supported functions with a single test body
   - Reduces duplication and ensures parity

5. **Add documentation example to public API:**
   - Show when pushdown is allowed vs. blocked
   - Helps users understand the feature

6. **Consider `NestedBehavior` enum for state clarity:**
   - Encode the three states: PrimitiveOnly, ListsSupported, Unsupported
   - Prevent subtle logic bugs in future changes

### Low Priority (Nice-to-Have)

7. **Document the function name detection assumption:**
   - Note that relying on `fun.name()` is pragmatic but not bulletproof
   - Suggest trait-based approach for future hardening

8. **Add comment explaining `ProjectionMask::leaves()` choice:**
   - Clarifies why leaf indices are used instead of root indices for nested columns

---

## Summary

**This commit successfully implements support for filtering nested list types during Parquet decoding.** The implementation is:

- ✅ **Functionally correct:** Logic properly validates pushdown eligibility and maps indices.
- ✅ **Well-tested:** Tests cover happy path, edge cases (structs), and end-to-end filtering.
- ✅ **Well-documented:** Module docs, rustdoc, and clear code.
- ⚠️ **Could be simplified:** Several refactoring suggestions would improve clarity without changing behavior.

**Recommendation:** **Approve and merge**. The non-blocking suggestions can be addressed in follow-up PRs or iterative refinement. The feature is production-ready and solves the stated problem comprehensively.

---

## Post-Merge Action Items (Optional)

1. Consider extracting `supports_list_predicates()` to a dedicated predicate registry module for extensibility.
2. Add parameterized tests for the three array functions.
3. Add documentation examples to public functions.
4. Explore trait-based function detection as a more robust alternative to name-based matching.
