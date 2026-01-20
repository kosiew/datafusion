# Issue: Audit `is_identity_assignment` for Qualifier Hazards (P3 - Safety Enhancement)

**Priority:** Low  
**Type:** Safety / Code Review  
**Effort:** Low to Medium  
**Timeline:** Follow-up work (post-PR #18840)  
**Related PR:** #18840 (Delete Filter Extraction from TableScan)  
**Reporter:** @ethan-tyler  
**Related Issues:** Similar to qualifier-stripping hazards in filter extraction

---

## Problem Statement

The `is_identity_assignment` function (used in UPDATE statement planning) likely performs qualifier stripping similar to what we found in `extract_dml_filters`. This creates the same class of safety hazards:

### Potential Issues

1. **Qualifier Collapse Risk**
   - Stripping qualifiers before validation may collapse distinct qualified assignments
   - Example: `UPDATE t SET t.col = t.col` vs. `UPDATE t SET source.col = t.col` 
   - After stripping: both become `col = col`, hiding semantic difference

2. **Cross-Table Assignment Contamination**
   - In UPDATE...FROM scenarios, assignments could reference wrong table columns
   - Example: `UPDATE target SET col = source.val FROM source` 
   - Without validation, could accidentally use `target.val` instead of `source.val`

3. **Silent Semantic Corruption**
   - Incorrect assignments don't raise errors, just produce wrong results
   - Much harder to debug than filter errors (no obvious query failure)

### Why This Matters

The `validate_and_strip_qualifiers` pattern we implemented for filter extraction (P2) should likely be applied to assignment validation as well. Identity assignments (like `SET col = col`) are commonly used to force row updates without changing values, so correctness is critical.

---

## Current Implementation (Presumed)

Based on the naming pattern and context, `is_identity_assignment` likely:

1. Takes an UPDATE assignment expression
2. Strips table qualifiers from both left and right sides
3. Compares the unqualified expressions
4. Returns `true` if they match (identity assignment)

**Presumed code pattern:**
```rust
fn is_identity_assignment(left: &Expr, right: &Expr) -> Result<bool> {
    let left_unqualified = strip_column_qualifiers(left)?;
    let right_unqualified = strip_column_qualifiers(right)?;
    Ok(left_unqualified == right_unqualified)
}
```

**The hazard:** If `left` and `right` reference different tables but have the same column name, stripping qualifiers makes them appear identical.

---

## Proposed Solution

Apply the same defense-in-depth pattern used for filter validation:

### Pattern 1: Validate Before Stripping

```rust
fn is_identity_assignment(
    left: &Expr, 
    right: &Expr,
    target_table_name: &str,
) -> Result<bool> {
    // Validate both sides reference only target table
    validate_assignment_columns(left, target_table_name)?;
    validate_assignment_columns(right, target_table_name)?;
    
    // Now safe to strip and compare
    let left_unqualified = strip_column_qualifiers(left.clone())?;
    let right_unqualified = strip_column_qualifiers(right.clone())?;
    
    Ok(left_unqualified == right_unqualified)
}

fn validate_assignment_columns(expr: &Expr, target_table_name: &str) -> Result<()> {
    let col_refs = expr.column_refs();
    
    for col_ref in col_refs {
        if let Some(qualifier) = &col_ref.relation {
            if qualifier.to_string() != target_table_name {
                return plan_err!(
                    "UPDATE assignment references column from non-target table: {}.{}. \
                     Only columns from table '{}' are allowed.",
                    qualifier,
                    col_ref.name,
                    target_table_name
                );
            }
        }
    }
    Ok(())
}
```

### Pattern 2: Context-Aware Comparison

Alternatively, compare WITH qualifiers first, only strip if necessary:

```rust
fn is_identity_assignment(
    left: &Expr,
    right: &Expr,
    target_table_name: &str,
) -> Result<bool> {
    // Try qualified comparison first (more precise)
    if left == right {
        return Ok(true);
    }
    
    // Fall back to unqualified comparison only if both reference target table
    if references_only_target(left, target_table_name)? 
        && references_only_target(right, target_table_name)? {
        let left_unqualified = strip_column_qualifiers(left.clone())?;
        let right_unqualified = strip_column_qualifiers(right.clone())?;
        return Ok(left_unqualified == right_unqualified);
    }
    
    // Different tables or invalid references
    Ok(false)
}
```

---

## Investigation Checklist

### Step 1: Locate the Function
- [ ] Search codebase for `is_identity_assignment` definition
- [ ] Identify file location (likely `datafusion/core/src/physical_planner.rs` or similar)
- [ ] Check if function exists or if similar logic is inline

### Step 2: Analyze Current Implementation
- [ ] Does it strip qualifiers?
- [ ] Does it validate column references before stripping?
- [ ] What happens with cross-table references?
- [ ] Are there any existing safety checks?

### Step 3: Check Call Sites
- [ ] Where is `is_identity_assignment` called?
- [ ] What context is available at call sites (e.g., target table name)?
- [ ] Are there any existing workarounds for cross-table scenarios?

### Step 4: Review Tests
- [ ] Are there tests for identity assignments?
- [ ] Do tests cover qualified vs. unqualified columns?
- [ ] Do tests cover cross-table reference cases?
- [ ] Any existing bug reports related to this?

### Step 5: Assess Impact
- [ ] How common is the affected code path?
- [ ] Are there known user-reported issues?
- [ ] Is UPDATE...FROM supported (makes this critical)?
- [ ] Performance impact of adding validation?

---

## Implementation Plan

### Phase 1: Discovery (1-2 days)
1. Locate `is_identity_assignment` or equivalent logic
2. Read and understand current implementation
3. Identify all call sites and usage patterns
4. Review related UPDATE tests

### Phase 2: Analysis (1-2 days)
1. Determine if hazard exists (does it strip without validating?)
2. Identify potential attack vectors or bug scenarios
3. Check if UPDATE...FROM makes this exploitable
4. Document findings

### Phase 3: Fix (if needed) (1-3 days)
1. Apply appropriate validation pattern from above
2. Add comprehensive tests for:
   - Identity assignments with qualifiers
   - Non-identity assignments with same column names
   - Cross-table reference attempts (if UPDATE...FROM exists)
3. Update error messages to be clear and helpful

### Phase 4: Verification (1 day)
1. All existing tests pass
2. New tests demonstrate correct behavior
3. No performance regression
4. Code review and documentation

---

## Test Cases to Add

### Test 1: Basic Identity Assignment
```sql
UPDATE t SET col = col WHERE id = 1
```
**Expected:** Recognized as identity assignment

### Test 2: Qualified Identity Assignment
```sql
UPDATE t SET t.col = t.col WHERE id = 1
```
**Expected:** Recognized as identity assignment

### Test 3: Mixed Qualification (Same Table)
```sql
UPDATE t SET col = t.col WHERE id = 1
```
**Expected:** Recognized as identity assignment (both reference same table)

### Test 4: Non-Identity Assignment
```sql
UPDATE t SET col1 = col2 WHERE id = 1
```
**Expected:** NOT recognized as identity assignment

### Test 5: Cross-Table Reference (if UPDATE...FROM supported)
```sql
UPDATE target SET col = source.col FROM source WHERE target.id = source.id
```
**Expected:** NOT recognized as identity assignment + error raised if source.col used incorrectly

### Test 6: Ambiguous Case (Future-Proofing)
```sql
UPDATE t SET t.col = other.col FROM other WHERE t.id = other.id
```
**Expected:** Error indicating cross-table reference detected

---

## Acceptance Criteria

- [ ] `is_identity_assignment` located and analyzed
- [ ] Qualifier hazard assessed (exists or doesn't exist)
- [ ] If hazard exists:
  - [ ] Validation function implemented following P2 pattern
  - [ ] All call sites updated to pass target table name
  - [ ] Comprehensive tests added
  - [ ] Error messages clear and actionable
- [ ] If no hazard:
  - [ ] Document why current implementation is safe
  - [ ] Add tests demonstrating safety
  - [ ] Mark issue as "verified safe"
- [ ] All tests pass
- [ ] Documentation updated

---

## Risks & Considerations

### Risk 1: No UPDATE...FROM Support
- **Impact:** Lower priority if multi-table UPDATEs don't exist
- **Mitigation:** Still fix for future-proofing and consistency

### Risk 2: Performance Overhead
- **Impact:** Validation adds column reference traversal
- **Mitigation:** Happens only during planning, not execution (minimal impact)

### Risk 3: Breaking Changes
- **Impact:** Stricter validation might reject previously-accepted queries
- **Mitigation:** Only reject semantically-incorrect queries (this is a feature)

### Risk 4: False Positive (Function Doesn't Exist)
- **Impact:** Spent effort on non-existent issue
- **Mitigation:** Quick discovery phase will identify this early

---

## Related Work

### Completed Prerequisites (PR #18840)
- ✅ P2: Qualifier-stripping validation for filters - provides the pattern to follow
- ✅ P2: Target scan scoping - establishes target table name availability
- ✅ `validate_and_strip_qualifiers` function - reusable validation pattern

### Parallel Concerns
- Identity assignment optimization (may rely on correct identity detection)
- UPDATE...FROM implementation (blocked by this safety audit)
- Assignment rewriting rules (may introduce new qualifier patterns)

---

## Success Metrics

- [ ] Code audit completed within 1 week
- [ ] If hazard found, fix implemented and tested within 2 weeks
- [ ] Zero regressions in existing UPDATE tests
- [ ] New tests provide >90% coverage of assignment validation paths
- [ ] Clear documentation of identity assignment semantics

---

## Open Questions

1. **Does `is_identity_assignment` exist?**
   - Need to search codebase to confirm
   - May be inline logic rather than named function

2. **What's the context at call sites?**
   - Is target table name available?
   - Are assignments already scoped to target table?

3. **Is there existing validation?**
   - Maybe the issue was already addressed
   - Check git history for related changes

4. **How common are identity assignments?**
   - Check user queries and benchmarks
   - Determines priority of optimization vs. safety

5. **Does UPDATE...FROM exist?**
   - If not, this is purely future-proofing
   - If yes, this is critical safety issue

---

## Example Scenarios

### Scenario 1: Forced Row Update
```sql
-- User wants to update timestamp without changing data
UPDATE users SET updated_at = NOW(), name = name WHERE id = 1;
```
**Expected:** `name = name` correctly identified as identity assignment, optimized away or preserved per update semantics.

### Scenario 2: Defensive Programming
```sql
-- User explicitly qualifies to be clear
UPDATE users SET users.email = users.email WHERE verified = false;
```
**Expected:** Qualifiers validated, then stripped; identity assignment recognized.

### Scenario 3: Typo Protection
```sql
-- User accidentally references wrong table (hypothetical UPDATE...FROM)
UPDATE target SET target.col = source.col FROM source WHERE target.id = source.id;
```
**Expected:** Error raised: "Assignment references non-target table column" (if validation exists).

---

## Implementation Locations (To Search)

### Primary Candidates
1. `datafusion/core/src/physical_planner.rs` - Main DML planning
2. `datafusion/sql/src/statement.rs` - SQL statement planning
3. `datafusion/expr/src/logical_plan/dml.rs` - DML logical plan structures
4. `datafusion/optimizer/` - Assignment rewriting rules

### Search Patterns
```bash
# Search for function name
git grep -n "is_identity_assignment"

# Search for identity assignment logic
git grep -n "identity.*assignment" 

# Search for UPDATE assignment handling
git grep -n "extract.*assignment"
git grep -n "UPDATE.*SET"

# Check for qualifier stripping in UPDATE paths
git grep -n "strip.*qualifier.*update" -i
```

---

## Documentation Needed

1. **Code Comments** - Document why validation is necessary
2. **Developer Guide** - Explain identity assignment semantics
3. **Test Documentation** - Describe test coverage for assignments
4. **Migration Notes** - If behavior changes for any edge cases

---

## Next Steps

1. **Immediate:** Search codebase for `is_identity_assignment` or equivalent logic
2. **Day 1-2:** Analyze implementation and assess hazard existence
3. **Day 3-5:** If hazard exists, implement fix following P2 pattern
4. **Day 6-7:** Add tests and verify correctness
5. **Final:** Document findings and close issue

---

## Conclusion

This audit is a **low-effort, high-value** safety check inspired by the qualifier hazards found in filter extraction. Even if no hazard exists, documenting the analysis provides confidence and serves as a reference for future similar audits.

The pattern established in P2 (validate before stripping) can be directly applied here, making implementation straightforward if needed. This is prudent engineering practice: apply the same safety standards consistently across similar code paths.

