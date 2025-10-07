# Min/Max Bytes Dense Inline Regression - Overview

## Observed Regression

Criterion detected statistically significant performance regressions after introducing the `DenseInline` mode optimization:

| Benchmark | Change | Impact |
|-----------|--------|--------|
| min bytes dense duplicate groups | **+20.02%** | ⚠️ Critical regression |
| min bytes dense first batch | **+1.60%** | ⚠️ Minor regression |
| min bytes dense reused accumulator | **+1.17%** | ⚠️ Minor regression |
| min bytes sparse groups | **-28.97%** | ✅ Major improvement |
| min bytes monotonic group ids | **-40.15%** | ✅ Major improvement |

All changes are statistically significant (p < 0.000001).

## Summary

While the `DenseInline` mode successfully eliminated quadratic scratch allocation overhead for sparse and monotonic workloads (achieving 29-40% improvements), it introduced 1-20% regressions in dense workload patterns. The root cause is **redundant statistics tracking**: the implementation collects per-batch density metrics to drive mode-switching heuristics, but these statistics are never used after the accumulator commits to `DenseInline` mode. For stable dense workloads, this tracking becomes pure overhead.

## Comprehensive Analysis

This regression has been thoroughly analyzed in the following documents:

1. **[Root Cause Analysis](./min_max_bytes_dense_inline_regression_root_cause.md)**
   - Detailed breakdown of why each benchmark regressed
   - Line-by-line code analysis with overhead calculations
   - Explanation of why improvements worked vs. why regressions occurred

2. **[Remediation Tasks](./min_max_bytes_dense_inline_regression_tasks.md)**
   - 6 prioritized tasks with implementation details
   - Code snippets and testing procedures
   - Expected impact for each fix

3. **[Summary Document](./min_max_bytes_dense_inline_regression_summary.md)**
   - Executive overview of the issue
   - Solution strategy and timeline
   - Success criteria and alternative approaches

## Quick Fix Summary

The primary fix is a **commit-once optimization**:

1. Track mode stability over first 2-3 batches
2. Once stable in `DenseInline` mode, set `dense_inline_committed = true`
3. Route committed batches to minimal fast path with no statistics tracking
4. Add run-length detection for consecutive duplicate groups
5. Defer mark allocation until second batch (optimize cold-start)

**Expected outcome**: Reduce regressions from +1-20% to ≤2% while maintaining 29-40% improvements.

## Related Files

- Implementation: `datafusion/functions-aggregate/src/min_max/min_max_bytes.rs`
- Benchmarks: `datafusion/functions-aggregate/benches/min_max_bytes.rs`
- Original issue: #17897 (quadratic scratch allocation problem)
- PR commits: `c1ac251d6^..93e1d7529`

## Next Steps

See **[Remediation Tasks](./min_max_bytes_dense_inline_regression_tasks.md)** for detailed implementation guidance.
