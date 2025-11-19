# HashMap-Based Optimization for requalify_sides_if_needed

This document contains an O(n+m) HashMap-based implementation of `requalify_sides_if_needed` as an alternative to the current O(n*m) nested loop implementation.

## Trade-off Analysis

**Current O(n*m) Implementation:**
- ✅ Simple, easy to understand
- ✅ Fast for typical small schemas (10-100 columns)
- ✅ Early return makes common case very fast
- ✅ No additional memory allocation
- ❌ Quadratic complexity for large schemas

**Proposed O(n+m) HashMap Implementation:**
- ✅ Better asymptotic complexity
- ✅ Scales well to hundreds of columns
- ❌ HashMap allocation overhead
- ❌ More complex code
- ❌ Need to handle multiple columns with same name (different qualifiers)
- ❌ Hashing overhead may not pay off for small schemas

## Recommended Implementation

```rust
use std::collections::HashMap;

/// Ensure that the left and right sides of a join have unique qualified field names.
/// If there are duplicates, we wrap each side in a `SubqueryAlias` with `"left"` and `"right"`.
/// Returns a tuple of `(left, right, was_requalified)`.
pub fn requalify_sides_if_needed(
    left: LogicalPlanBuilder,
    right: LogicalPlanBuilder,
) -> Result<(LogicalPlanBuilder, LogicalPlanBuilder, bool)> {
    let left_cols = left.schema().columns();
    let right_cols = right.schema().columns();

    // Build HashMap of left columns grouped by name for O(1) lookup
    // Use Vec to handle multiple columns with same name but different qualifiers
    let mut left_by_name: HashMap<&str, Vec<&Column>> = HashMap::new();
    for col in &left_cols {
        left_by_name.entry(col.name.as_str()).or_default().push(col);
    }

    // Check right columns against left - O(m) with O(1) lookups
    for r in &right_cols {
        if let Some(left_matches) = left_by_name.get(r.name.as_str()) {
            // Found columns with matching names - check for conflicts
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

## Performance Characteristics

### Current Implementation (Nested Loop)
- **Best case (conflict found early):** O(1) - immediate return
- **Average case (conflict in middle):** O(n*m/2)
- **Worst case (no conflict):** O(n*m)
- **Space:** O(1) - no additional allocation
- **For n=m=50:** ~2,500 iterations worst case
- **For n=m=100:** ~10,000 iterations worst case

### HashMap Implementation
- **Best case:** O(n+m) - always must build HashMap
- **Average case:** O(n+m)
- **Worst case:** O(n+m)
- **Space:** O(n) - HashMap with n entries
- **For n=m=50:** ~100 operations + HashMap overhead
- **For n=m=100:** ~200 operations + HashMap overhead

## Benchmark Scenarios

**Small schemas (10-20 columns) - TYPICAL:**
- Nested loop: ~100-400 comparisons worst case
- HashMap: ~20-40 operations + allocation overhead
- **Winner:** Nested loop (simpler, less overhead)

**Medium schemas (50-100 columns) - COMMON:**
- Nested loop: ~2,500-10,000 comparisons worst case
- HashMap: ~100-200 operations + allocation overhead
- **Winner:** Depends on conflict location; probably similar performance

**Large schemas (200+ columns) - RARE:**
- Nested loop: ~40,000+ comparisons worst case
- HashMap: ~400+ operations + allocation overhead
- **Winner:** HashMap (significantly better)

## Recommendation

**Keep the current nested loop implementation** because:

1. **Typical use case favors simplicity:** Most DataFusion schemas have 10-100 columns
2. **Early return optimization:** Common case (conflicts exist) exits very quickly
3. **Not a hot path:** Only called during plan construction, not execution
4. **Code clarity:** Easier to understand and maintain
5. **Follows project guidelines:** AGENTS.md emphasizes simplicity unless bottleneck proven

**Consider HashMap optimization if:**
- Profiling shows this function as a bottleneck
- Users commonly work with schemas of 200+ columns
- There's evidence of performance issues in production

## Testing Both Implementations

If desired, we could add a feature flag or configuration option to allow switching between implementations for benchmarking:

```rust
#[cfg(feature = "optimize-requalify")]
pub fn requalify_sides_if_needed(...) {
    // HashMap-based O(n+m) implementation
}

#[cfg(not(feature = "optimize-requalify"))]
pub fn requalify_sides_if_needed(...) {
    // Current O(n*m) implementation
}
```

This would allow real-world performance comparison on actual workloads.
