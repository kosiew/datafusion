# PR Review: Nested DataType Filter Pushdown to Parquet

**Commit Range:** `b64838e4c^..ce3a1b8f9`  
**Review Date:** December 29, 2025

---

## Summary

This PR implements filter pushdown support for nested data types (specifically list columns) in Parquet file decoding. The implementation adds a trait-based system for identifying supported array predicates (`array_has`, `array_has_all`, `array_has_any`, and NULL checks) and routes them to the Parquet decoder level, while maintaining the existing behavior for unsupported nested types like structs.

---

## Decision

✅ **Approve with suggestions**

The implementation is functionally correct and addresses the feature request. Code quality is high, with comprehensive tests and clear documentation. However, there are some non-blocking improvements around code organization, naming clarity, and minor ergonomic enhancements.

---

## Detailed Review

### 1. **Design & Architecture** ✅

#### Strengths

- **Trait-based approach:** The `SupportsListPushdown` trait is a clean, extensible design that allows new supported predicates to be registered without modifying predicate evaluation logic.
- **Explicit state tracking:** The `NestedBehavior` enum (`PrimitiveOnly`, `ListsSupported`, `Unsupported`) makes the implicit states in the original code explicit and easier to reason about.
- **Separation of concerns:** Logic is properly split across modules:
  - `supported_predicates.rs` — registry of which expressions support list pushdown
  - `row_filter.rs` — coordinate system tracking (root vs. leaf indices) and filter building

#### Suggestions

1. **Consider naming:** `NestedBehavior` could be more specific, e.g., `NestedColumnSupport` or `PushdownNestedType` to clarify this describes *support for nested types*, not generic nested behavior.

2. **Root-to-leaf mapping:** The `leaf_indices_for_roots` function performs an important transformation but its semantics could be clearer in documentation:
   - For primitives: 1:1 mapping
   - For nested: 1:N expansion to all leaf columns under roots
   
   Consider a doc comment with an ASCII diagram:
   ```rust
   /// Maps root column indices to leaf column indices in the Parquet schema.
   ///
   /// For primitive columns:
   ///   Root 1 (primitive) → Leaf 1
   ///
   /// For nested columns (e.g., structs with fields):
   ///   Root 2 (struct)    → Leaves 2, 3, 4  (all fields under struct)
   ///   Root 5 (list)      → Leaf 5          (list is a single leaf)
   ```

---

### 2. **Code Quality & Clarity** ✅

#### Strengths

- **Well-documented public API:** Module-level docs in `row_filter.rs` are comprehensive, including semantics and examples.
- **Clear variable names:** `non_primitive_columns`, `allow_list_columns`, `leaf_indices` are self-documenting.
- **Proper error handling:** Uses `Result<T>` and propagates errors with `?`.

#### Observations

1. **PushdownChecker complexity:** The `check_single_column` method contains a nested conditional that could benefit from extraction:

   ```rust
   // Current (lines 295-325)
   fn check_single_column(&mut self, column_name: &str) -> Option<TreeNodeRecursion> {
       if let Ok(idx) = self.file_schema.index_of(column_name) {
           self.required_columns.insert(idx);
           if DataType::is_nested(self.file_schema.field(idx).data_type()) {
               let is_list = matches!(...);
               let is_supported = self.allow_list_columns && is_list;
               if is_supported { /* ... */ } else { /* ... */ }
           }
       } else {
           self.projected_columns = true;
       }
       None
   }
   ```

   **Suggestion:** Extract the nested type check into a helper:
   ```rust
   fn is_nested_type_supported(&self, dt: &DataType) -> bool {
       let is_list = matches!(dt, DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _));
       self.allow_list_columns && is_list
   }
   ```

2. **BTreeSet → Vec conversion:** In `FilterCandidateBuilder::build`, converting `required_columns` to `Vec` is straightforward, but for very large schemas, maintaining iteration order may matter. Currently this is fine, but a comment explaining ordering doesn't matter here would be helpful.

---

### 3. **API Design & Consistency** ✅

#### Strengths

- **Matches ecosystem patterns:** Consistent with how `ScalarFunctionExpr`, `Column`, and other physical expressions are handled.
- **No breaking changes:** Existing behavior is preserved for unsupported types.

#### Observations

1. **ProjectionColumns struct:** After refactoring in commit `14fc2e318`, the struct only contains `leaf_indices`. The struct is still useful (encapsulation), but the name could be more specific:

   ```rust
   // Current
   struct ProjectionColumns {
       leaf_indices: Vec<usize>,
   }
   ```

   **Alternative names considered:**
   - `LeafProjection` — directly states what it contains
   - `ParquetColumnProjection` — clarifies scope (Parquet leaf semantics)
   - `FilterColumnProjection` — context-aware

   **Recommendation:** If this struct might grow in the future (e.g., to track nullable/non-nullable splits), the current name is fine. If it's purely a wrapper for `leaf_indices`, consider inlining it into `FilterCandidate` as a direct field: `leaf_indices: Vec<usize>`.

2. **Function visibility:** `supports_list_predicates` is internal (`super::supported_predicates::`). Consider whether it should be public for testing or external analyzer tools. Currently, this is appropriate — implementation detail.

---

### 4. **Testing** ✅✅

#### Strengths

- **Comprehensive test suite:** 250+ lines of integration tests in `filter_pushdown.rs` covering:
  - `array_has()` pushdown verification
  - `array_has_all()` with multiple predicates
  - `array_has_any()` variants
  - Conjunction with other predicates (`id > 1 AND array_has(...)`)
  - Disjunction (OR) cases

- **Test structure:** Helper functions like `create_test_parquet_file()` and `assert_predicate_pushed_down()` reduce boilerplate.

- **Unit tests in `supported_predicates.rs`:** Cover null-check detection and scalar function matching.

#### Suggestions

1. **Edge case coverage:** Consider adding tests for:
   - Negative cases: struct columns with array predicates (should NOT pushdown)
   - Mixed predicates: `(array_has(tags, 'x') AND struct_field IS NOT NULL)` — should NOT pushdown
   - Deeply nested lists: `array_has(list_of_lists, ...)`
   - Null array column filters: `array IS NULL` (should pushdown)

2. **Test organization:** Current tests in the module are organized but could benefit from doc comments explaining what each `#[tokio::test]` validates.

3. **No snapshot tests:** Good choice to avoid `.snap` files; SLT (SQL Logic Tests) would be even better for reproducibility, but the current integration tests are acceptable.

---

### 5. **Consistency with Codebase** ✅

#### Matches patterns in AGENTS.md

- ✅ Uses `Result<T>` with `?` operator for error propagation
- ✅ Follows naming conventions: `snake_case` functions, `CamelCase` types
- ✅ Leverages existing helpers like `TreeNodeVisitor` for traversal
- ✅ Proper use of `Arc<dyn PhysicalExpr>` for expression sharing
- ✅ Functions kept under 40 lines (mostly; `check_single_column` is ~30 lines with nesting)

#### Matches patterns in copilot-instructions.md

- ✅ Two-layer expression API: Proper distinction between logical and physical expressions
- ✅ Extensibility traits: `SupportsListPushdown` follows the extensibility pattern
- ✅ Performance-aware: Zero-copy semantics preserved; no unnecessary allocations

---

### 6. **Potential Issues & Edge Cases**

#### No blocking issues found, but consider:

1. **Arrow-rs dependency on nested filtering:** The PR assumes arrow-rs correctly handles nested list column filtering via `ProjectionMask::leaves()`. This is a hard dependency on arrow-rs behavior. Should this assumption be documented or tested?
   - **Mitigation:** Document in module-level comment that arrow-rs must support list column projection.

2. **Parquet schema assumptions:** `leaf_indices_for_roots` assumes Parquet schema structure matches Arrow's understanding of leaf vs. root. This is generally safe but sensitive to schema evolution.
   - **Recommendation:** Add a comment: "Assumes Parquet schema descriptor structure matches Arrow's field layout."

3. **List type variants:** The code explicitly handles `List`, `LargeList`, and `FixedSizeList` separately. This is correct, but if new list types are added to Arrow, this code won't auto-detect them.
   - **Recommendation:** No action needed, but consider a TODO comment for future maintainers.

---

### 7. **Performance Considerations** ✅

#### Strengths

- **No performance regressions:** The predicate checking (via `supports_list_predicates`) is O(n tree depth), which is negligible compared to actual Parquet I/O.
- **Early exit optimization:** `TreeNodeRecursion::Jump` prevents unnecessary traversal of subtrees when unsupported columns are found.
- **Pre-computation of indices:** Leaf indices are computed once during `FilterCandidate` construction, not per-batch.

#### Observations

- **Conversion overhead:** `BTreeSet → Vec` conversion in `FilterCandidateBuilder::build` is O(n) but n is typically small (~1-10 columns per filter).

---

### 8. **Documentation** ✅✅

#### Strengths

- **Module-level documentation:** Excellent overview of pushdown semantics and list-aware predicate support.
- **Examples in `can_expr_be_pushed_down_with_schemas` doc comments:** Clear usage patterns.
- **Enum documentation:** `NestedBehavior` variants are well-explained.

#### Suggestions

1. **Add examples to `leaf_indices_for_roots`:**
   ```rust
   /// # Examples
   /// ```ignore
   /// // For primitive columns (id=0, age=1):
   /// // root_indices = [0, 1] → leaf_indices = [0, 1]
   /// 
   /// // For a struct root (id=0, person=1 with fields name, age):
   /// // root_indices = [1] → leaf_indices = [1, 2, 3]
   /// ```
   ```

2. **Document limitations:** Add a comment about what *cannot* be pushed down:
   ```rust
   /// Note: Struct columns and other complex nested types are not supported
   /// for pushdown. They will be evaluated post-decode.
   ```

---

### 9. **Scope & Requirements** ✅

The PR successfully addresses the feature request:

- ✅ Implements support for filtering nested data types (lists)
- ✅ Supports useful operations: `array_has`, `array_has_all`, `array_has_any`
- ✅ Maintains backward compatibility (structs still evaluated post-decode)
- ✅ Demonstrates Arrow compatibility (no errors in tests)
- ✅ No scope creep; changes are focused on Parquet row filter mechanism

---

### 10. **Commit Organization** ✅

The commits in the range show a well-organized development process:

1. **b64838e4c:** Document nested pushdown semantics (foundation docs)
2. **14fc2e318–8dceb20df:** Refactor row_filter.rs with trait-based approach
3. **92b92ca39–0e72d700e:** Add comprehensive array function tests
4. **7c70b9cd4–c8394eb28:** Consolidate and refine tests
5. **644ebb2e5–0e293a7eb:** Example code (then reverted)
6. **ce3a1b8f9:** Final test commit for integration

**Observation:** The revert of the array functions example (644ebb2e5 → 0e293a7eb) suggests the maintainer felt examples belonged elsewhere or would be added in a follow-up. This is acceptable.

---

## Summary of Suggestions

| Category | Priority | Suggestion |
|----------|----------|-----------|
| **Naming** | Low | Rename `NestedBehavior` to something more specific (e.g., `NestedColumnSupport`) |
| **Code Organization** | Low | Extract nested type support check into `is_nested_type_supported()` helper |
| **Documentation** | Low | Add ASCII diagram to `leaf_indices_for_roots` explaining 1:1 vs. 1:N mapping |
| **Testing** | Low | Add negative case tests (struct with array predicate should NOT pushdown) |
| **ProjectionColumns** | Very Low | Consider if struct can be inlined into `FilterCandidate` if it only holds `leaf_indices` |

---

## Approval Rationale

✅ **This PR is ready to merge.**

- **Correctness:** Implementation correctly identifies supported list predicates and routes them to Parquet decoder. No logic errors detected.
- **Tests:** Comprehensive test coverage including physical plan verification.
- **Documentation:** Well-documented with clear examples and module-level overview.
- **Backward Compatibility:** Struct columns and other unsupported types retain post-decode evaluation behavior.
- **Performance:** No regressions; predicate checking is efficient.
- **Code Quality:** Idiomatic Rust, consistent with DataFusion conventions.

The suggestions above are quality-of-life improvements for future maintainers and do not block this PR.

---

## Recommendation for Follow-up

If this feature sees adoption, consider:

1. **Extending supported functions:** Evaluate whether `array_contains`, `array_concat`, or other array functions can be safely pushed down.
2. **SQL Logic Tests:** Consider migrating array predicate pushdown tests to `datafusion/sqllogictest/test_files/` for easier maintenance.
3. **Arrow-rs collaboration:** If more nested types become relevant, coordinate with arrow-rs maintainers on Parquet decoding support.

---

**Reviewer:** GitHub Copilot (Claude Haiku 4.5)  
**Date:** December 29, 2025
