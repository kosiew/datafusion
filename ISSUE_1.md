# Issue: Unified `TableScan.filters` Design (P3 - Architectural Enhancement)

**Priority:** Very High  
**Type:** Architecture / Enhancement  
**Effort:** Large (RFC + Implementation)  
**Timeline:** Future enhancement (post-PR #18840)  
**Related PR:** #18840 (Delete Filter Extraction from TableScan)  
**Reporter:** @adriangb

---

## Problem Statement

The current DML filter extraction implementation in DataFusion exhibits a **fundamental architectural inconsistency**: the `TableScan` logical plan node maintains filters in two distinct ways:

1. **Filters passed to TableProvider** - Via `delete_from()`, `update()`, etc., operators
2. **Filters in LogicalPlan** - Via `TableScan.filters` field during planning

This dual-track approach creates several problems:

### Current Issues

1. **Filter Duplication Risk**
   - Same predicate may exist in both `Filter` node and `TableScan.filters`
   - Deduplication logic becomes complex and error-prone
   - Different code paths may process the same filter inconsistently

2. **Semantic Confusion**
   - Unclear which filters are "pushed down" vs. "logical"
   - Makes it difficult to reason about query semantics
   - Complicates optimizer correctness proofs

3. **Implementation Burden**
   - DML operations must collect filters from multiple locations
   - Qualifier-stripping and validation needed at collection time
   - Each new operator/optimizer rule must handle both locations

4. **Multi-Table Safety Hazards**
   - UPDATE...FROM scenarios require careful tracking of which table each filter belongs to
   - Target table scoping becomes a bandage rather than systemic solution
   - Cross-table predicate contamination possible if new patterns emerge

### Example Scenario (UPDATE...FROM)

```sql
UPDATE target SET col = val 
FROM source 
WHERE target.id = source.id AND target.status = 'active'
```

**Current fragility:**
- Filters could appear in: Filter node, target TableScan.filters, or source TableScan.filters
- DML must track: which filters belong to target vs. source
- Risk: Source filter accidentally applied to target during extraction

**With unified design:**
- Single, clear filter representation
- Optimizer ensures filters on each table are stored consistently
- DML extraction becomes straightforward and safe

---

## Proposed Solution: Unified Filter Field

### Design Goals

1. **Single Source of Truth** - One representation of filters for each table
2. **Explicit Semantics** - Clear distinction between logical and pushed-down filters
3. **Safety** - Impossible for filters to be duplicated or cross-contaminated
4. **Performance** - No additional overhead vs. current approach

### Approach

Introduce a **unified filter contract** where:

1. **TableScan Becomes the Filter Container**
   - `TableScan.filters` becomes the *only* place where table predicates live during planning
   - Filter nodes are removed/consolidated during logical planning (not pushed down later)

2. **Filter Node Redesign**
   - Retains Filter node for predicates that can't be expressed in TableScan
   - Examples: complex expressions, cross-table joins, subqueries
   - Simple table-local predicates live in TableScan.filters

3. **Optimizer Clarity**
   - "Push down" means: move predicate from Filter into TableScan.filters
   - "Pull up" means: move predicate from TableScan.filters into Filter (rare)
   - Symmetry and clarity in optimizer rules

### Benefits

| Benefit | Current | Unified | Impact |
|---------|---------|---------|--------|
| Filter extraction | Multiple locations | Single location | Simpler DML logic |
| Cross-table safety | Complex tracking | Scoped by design | Safer UPDATE...FROM |
| Deduplication | Needed at runtime | Impossible structurally | Fewer bugs |
| Optimizer rules | Must handle both | Single representation | Cleaner code |
| Reasoning about plans | Unclear semantics | Explicit semantics | Better debugging |

---

## Implementation Path

### Phase 1: Design & RFC
- [ ] Create RFC document with detailed semantics
- [ ] Discuss with maintainers (especially @adriangb)
- [ ] Get community feedback on design choices
- [ ] Finalize filter representation strategy

### Phase 2: Core Infrastructure
- [ ] Implement new filter representation in LogicalPlan
- [ ] Update logical planner to use unified representation
- [ ] Modify optimizer rules to work with new representation
- [ ] Add comprehensive tests for new semantics

### Phase 3: Operator Updates
- [ ] Update all execution operators to use new filter source
- [ ] Verify filter semantics in physical plans
- [ ] Update metrics and profiling for filter tracking

### Phase 4: Migration
- [ ] Update existing code to use unified design
- [ ] Deprecate old dual-track filter handling
- [ ] Update documentation and examples

### Phase 5: Verification
- [ ] Full test suite passes
- [ ] Performance benchmarks unchanged or improved
- [ ] SQL logic tests pass
- [ ] TPC-DS/TPC-H benchmarks pass

---

## Acceptance Criteria

- [ ] RFC completed and reviewed
- [ ] Single, unified filter representation in TableScan
- [ ] No duplicate filters in logical plans
- [ ] DML filter extraction simplified (no multi-location collection needed)
- [ ] UPDATE...FROM works safely without cross-table filter contamination
- [ ] All existing tests pass
- [ ] Performance metrics maintained or improved
- [ ] New design documented in developer guide

---

## Related Work

### Immediate Prerequisites (Completed in PR #18840)
- ✅ P0: Explicit variant handling - ensures new plan types are visible to DML
- ✅ P1: UPDATE test coverage - locks in DML semantics
- ✅ P2: Target scan scoping - prevents cross-table filter extraction
- ✅ P2: Qualifier-stripping validation - validates filter column references

### Parallel Work (Not Blocked)
- P3: Audit `is_identity_assignment` - similar qualification concerns in UPDATE assignments

---

## Alternatives Considered

### Alternative 1: Keep Dual-Track, Improve Deduplication
- **Pros:** Minimal breaking changes
- **Cons:** Doesn't address fundamental semantic confusion
- **Decision:** Rejected - addresses symptoms, not root cause

### Alternative 2: Always Push Down (Remove Filter Nodes Early)
- **Pros:** Simpler structure
- **Cons:** May prevent some optimizations that reason about Filter nodes
- **Decision:** Rejected - might regress on optimization opportunities

### Alternative 3: Always Keep Filters Above TableScan
- **Pros:** Preserves current logical plan structure
- **Cons:** Complicates push-down tracking, doesn't solve cross-table issues
- **Decision:** Rejected - doesn't address safety concerns

### Selected: Approach 1 (Unified TableScan.filters)
- **Rationale:** Explicit semantics, safety by design, cleaner code
- **Best fit:** Matches DataFusion's philosophy of "push down what you can"

---

## Risks & Mitigation

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Large refactor affecting many operators | High | Phased implementation, comprehensive testing |
| Performance regression if not careful | Medium | Early benchmarking, incremental verification |
| Community resistance to changes | Low | RFC discussion, clear benefits documentation |
| Third-party code using old semantics | Medium | Long deprecation period, migration guide |

---

## Testing Strategy

### Unit Tests
- Logical planner produces unified filter representation
- Optimizer rules work with new representation
- Filters don't duplicate across locations

### Integration Tests
- Full DML pipelines (DELETE, UPDATE) work correctly
- UPDATE...FROM with multiple tables filters correctly
- Filter pushdown and pull-up work as designed

### Regression Tests
- All existing SQL logic tests pass
- TPC-DS/TPC-H benchmarks pass
- Performance metrics maintained

### Fuzzing
- Query fuzzer validates filter semantics
- Comparative tests vs. current implementation

---

## Documentation Needed

1. **Developer Guide** - Explanation of unified filter design
2. **Contributor Guide** - How to add new filters/operators
3. **Migration Guide** - If any third-party code needs updates
4. **RFC** - Detailed technical specification

---

## Questions for Discussion

1. Should Filter nodes be completely removed, or kept for non-pushable predicates?
2. How should cross-table predicates be represented? (Keep in Filter node?)
3. What's the story for subquery filters and correlated predicates?
4. Performance implications of always materializing TableScan.filters?
5. How does this interact with materialized view pushdown?

---

## Implementation Notes

### Key Files to Modify
- `datafusion/expr/src/logical_plan/` - LogicalPlan structures
- `datafusion/optimizer/` - Filter push-down rules
- `datafusion/physical-planner/` - Physical plan generation
- `datafusion/physical-plan/` - Execution operators
- `datafusion/core/src/physical_planner.rs` - DML planning

### No Breaking Changes If
- We maintain filter collection APIs for backward compatibility
- We provide migration utilities for third-party code
- We have a deprecation period before removing old code paths

---

## Success Metrics

- [ ] Lines of code in filter-handling reduced by 20%+
- [ ] DML filter extraction simplified (fewer edge cases)
- [ ] UPDATE...FROM implementation straightforward (no special scoping needed)
- [ ] Query optimizer reasoning clearer and more composable
- [ ] Zero performance regression on TPC-DS/H benchmarks
- [ ] Developer satisfaction survey shows improved understanding

---

## Timeline Estimate

**Assuming full-time focus:**
- RFC & Design: 1-2 weeks
- Phase 1-2 (Core): 3-4 weeks
- Phase 3-4 (Migration): 2-3 weeks
- Phase 5 (Verification): 1-2 weeks

**Total: 7-11 weeks** (likely to extend due to community review and iteration)

**Realistic timeline with part-time contributors: 3-6 months**

---

## Conclusion

The unified `TableScan.filters` design is a significant architectural improvement that:

1. **Solves immediate safety concerns** around multi-table DML operations
2. **Simplifies optimizer reasoning** about filter semantics
3. **Reduces implementation burden** for new operators and rules
4. **Enables confidence** in UPDATE...FROM and similar complex operations

While large in scope, this enhancement addresses a fundamental architectural issue that will continue to cause subtle bugs and complexity if left unaddressed. The investment in this RFC and implementation will pay dividends in code clarity, maintainability, and correctness.

---

## Next Steps

1. **Social:** Share this issue with @adriangb and other core maintainers
2. **RFC:** Create detailed RFC document with examples and open for discussion
3. **Planning:** Get community consensus on approach and timeline
4. **Implementation:** Begin Phase 1 design work

