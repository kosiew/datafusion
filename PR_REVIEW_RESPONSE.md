# PR Review Response

## Summary of Changes

This PR fixes the issue where Substrait round-trip mode fails on INTERSECT/EXCEPT queries with self-referential tables. The core problem was that when both inputs to a set operation reference the same table, the internal join logic would attempt to merge two schemas with identical qualified field names, causing a "Schema contains duplicate qualified field name" error.

The fix involves:
1. Calling `requalify_sides_if_needed()` in the `intersect_or_except()` function to automatically detect and resolve naming conflicts
2. Enhancing the conflict detection logic in `requalify_sides_if_needed()` to handle three conflict scenarios comprehensively
3. Adding comprehensive test coverage for self-referential INTERSECT and EXCEPT operations (both with and without ALL)

---

## Response to Review Comments

### Comment 1: Difference between INTERSECT and EXCEPT plans

**Reviewer:** martin-g  
**File:** `datafusion/substrait/tests/cases/roundtrip_logical_plan.rs`  
**Comment:**
> Is there a difference between the plans for INTERSECT (`self_referential_intersect`) and EXCEPT (`self_referential_except`)?
> I don't see any.

**Response:**

You're absolutely correct—the expected plans look identical in the test assertions, which is confusing. The key difference is actually in the **join type**, not the overall structure:

- `self_referential_intersect` produces: `**LeftSemi** Join: left.a = right.a`
- `self_referential_except` produces: `**LeftAnti** Join: left.a = right.a`

The rest of the plan structure is identical because:
1. Both operate on the same table (`data`) with similar filters
2. Both include the DISTINCT operation (via `Aggregate: groupBy=[[data.a]]`) because neither uses `ALL`
3. Both get requalified to `left` and `right` aliases due to the duplicate field name issue

**Proposed Action:**

I should update the test comments to explicitly highlight the join type difference. For example:

```rust
#[tokio::test]
async fn self_referential_intersect() -> Result<()> {
    // Test INTERSECT with the same table on both sides
    // This previously failed with "Schema contains duplicate qualified field name"
    // The fix ensures requalify_sides_if_needed is called in intersect_or_except
    // 
    // Key difference from EXCEPT: Uses **LeftSemi** join (returns rows from left that exist in right)
    // After roundtrip through Substrait, SubqueryAlias is lost and requalification
    // produces "left" and "right" aliases
    ...
}

#[tokio::test]
async fn self_referential_except() -> Result<()> {
    // Test EXCEPT with the same table on both sides
    // This previously failed with "Schema contains duplicate qualified field name"
    // The fix ensures requalify_sides_if_needed is called in intersect_or_except
    //
    // Key difference from INTERSECT: Uses **LeftAnti** join (returns rows from left that don't exist in right)
    // After roundtrip through Substrait, SubqueryAlias is lost and requalification
    // produces "left" and "right" aliases
    ...
}
```

Additionally, I could add assertions that explicitly verify the join type to make the tests more robust and self-documenting.

---

### Comment 2: O(n*m) complexity optimization

**Reviewer:** martin-g  
**File:** `datafusion/expr/src/logical_plan/builder.rs`  
**Comment:**
> Here the complexity is **O(n*m)**.
> You could optimize it to **O(n+m)** by iterating over `left_cols` (O(n)) and storing them in a `HashMap<ColumnName, Column>`, then iterating over `right_cols` (O(m)) with lookup by name in the hashmap (O(1)), and doing the checks when there is an entry for that name.

**Response:**

Excellent observation on the algorithmic complexity. You're correct that the current nested loop is O(n*m), and this can be optimized to O(n+m) using a HashMap.

**Analysis:**

However, I would like to offer some context on whether this optimization is **necessary in practice**:

1. **Schema size is typically small:** In most real-world queries, the number of columns in a schema is relatively small (10-100 columns). Even with 100 columns on each side, we're looking at 10,000 iterations, which is negligible for modern CPUs.

2. **Early return on conflict:** The function returns immediately upon finding the first conflict, so in the common case where conflicts exist (which is when this function matters), we often exit very early in the iteration.

3. **Simple conflict detection logic:** The current implementation is straightforward and easy to reason about. The match statement clearly shows all conflict scenarios.

4. **Called infrequently:** This function is only called during logical plan construction, not during execution. It's not in a hot path that runs millions of times.

**Trade-offs of HashMap approach:**

**Pros:**
- Better asymptotic complexity: O(n+m) vs O(n*m)
- Scales better for schemas with hundreds of columns

**Cons:**
- More memory allocation overhead for the HashMap
- More complex code that's slightly harder to understand
- HashMap construction and hashing overhead may not pay off for small schemas
- Need to handle the case where multiple columns have the same name in one schema (which can happen with different qualifiers)

**Recommendation:**

Given DataFusion's principles from `AGENTS.md`:

> **Performance Considerations:**
> - Optimizations should be focused on bottlenecks — those steps that are repeated millions of times in a query; otherwise, prefer simplicity.

I would recommend **keeping the current O(n*m) implementation** unless profiling shows this is actually a bottleneck. The simplicity and clarity of the current code align better with the codebase's maintainability goals.

**However**, if the reviewer feels strongly about this or if we anticipate very wide schemas (hundreds of columns), I'm happy to implement the HashMap-based optimization. Here's a sketch of what it would look like:

```rust
pub fn requalify_sides_if_needed(
    left: LogicalPlanBuilder,
    right: LogicalPlanBuilder,
) -> Result<(LogicalPlanBuilder, LogicalPlanBuilder, bool)> {
    let left_cols = left.schema().columns();
    let right_cols = right.schema().columns();

    // Build HashMap of left columns by name for O(1) lookup
    let mut left_by_name: HashMap<&str, Vec<&Column>> = HashMap::new();
    for col in &left_cols {
        left_by_name.entry(&col.name).or_default().push(col);
    }

    // Check right columns against left
    for r in &right_cols {
        if let Some(left_matches) = left_by_name.get(r.name.as_str()) {
            for l in left_matches {
                // Same name - check if this would cause a conflict
                match (&l.relation, &r.relation) {
                    // Both qualified with same relation - duplicate qualified field
                    (Some(l_rel), Some(r_rel)) if l_rel == r_rel => {
                        return Ok((
                            left.alias(TableReference::bare("left"))?,
                            right.alias(TableReference::bare("right"))?,
                            true,
                        ));
                    }
                    // Both unqualified - duplicate unqualified field
                    (None, None) => {
                        return Ok((
                            left.alias(TableReference::bare("left"))?,
                            right.alias(TableReference::bare("right"))?,
                            true,
                        ));
                    }
                    // One qualified, one not - ambiguous reference
                    (Some(_), None) | (None, Some(_)) => {
                        return Ok((
                            left.alias(TableReference::bare("left"))?,
                            right.alias(TableReference::bare("right"))?,
                            true,
                        ));
                    }
                    // Different qualifiers - OK, no conflict
                    _ => {}
                }
            }
        }
    }

    // No conflicts found
    Ok((left, right, false))
}
```

**Question for the reviewer:**

Would you like me to implement the HashMap-based optimization, or would you prefer to keep the simpler implementation given the typical small schema sizes and infrequent call pattern?

---

## Additional Notes

### Test Coverage

The PR includes comprehensive test coverage:
- `self_referential_intersect` - Tests INTERSECT without ALL (includes DISTINCT)
- `self_referential_except` - Tests EXCEPT without ALL (includes DISTINCT)
- `self_referential_intersect_all` - Tests INTERSECT ALL (no DISTINCT)
- `self_referential_except_all` - Tests EXCEPT ALL (no DISTINCT)

All tests verify the Substrait round-trip behavior and ensure the requalification works correctly.

### Conflict Detection Enhancement

The enhanced `requalify_sides_if_needed()` function now explicitly handles three conflict scenarios:
1. **Duplicate qualified fields:** Both sides have the same `relation.name` combination
2. **Duplicate unqualified fields:** Both sides have the same unqualified column name
3. **Ambiguous reference:** One side qualified, the other unqualified, but same name

This is more robust than the previous simple equality check.

---

## Action Items

Based on the review feedback:

1. **For Comment 1 (Plan differences):**
   - [ ] Update test comments to explicitly highlight the LeftSemi vs LeftAnti join type difference
   - [ ] Consider adding explicit assertions on join type if beneficial

2. **For Comment 2 (Complexity optimization):**
   - [ ] Await reviewer's decision on whether to implement HashMap-based O(n+m) optimization
   - [ ] If requested, implement the optimization while maintaining code clarity
   - [ ] Add comments explaining the algorithmic choice

Please let me know your preferences, and I'll implement the requested changes.
