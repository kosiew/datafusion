# PR Review Responses

## 1) Comment on CASE-heavy benchmark shape (`datafusion/core/benches/sql_planner_extended.rs`)

**Reviewer comment**
> I don't think we really want a benchmark for the case expression.  
> We want to optimize for the evaluation cost of the filter during pushdown, so perhaps it could be written not using a large case expression as is done currently or adaptive removing filters, etc.

**Proposed response**
Thanks, this is a fair concern. The CASE-heavy shape was added intentionally to reproduce the profiler hotspot we observed in `PushDownFilter` (nullability/type inference around non-inner joins), not as a claim that CASE itself is the primary real-world workload.

To make this clearer and avoid overfitting, I will treat the CASE query as a targeted micro-benchmark for one expensive planner path, and pair it with a simpler non-CASE LEFT JOIN filter shape so we can distinguish:

1. generic pushdown/filter-planning cost, and  
2. additional overhead from CASE/nullability inference.

This keeps the benchmark actionable for optimization work while preserving broader relevance.

**Plan**
1. Keep the existing CASE-heavy benchmark but rename/label it explicitly as a hotspot micro-benchmark.  
2. Add a companion non-CASE LEFT JOIN benchmark variant (same sweep dimensions) as a control.  
3. Report both variants in benchmark results so optimizations can be validated against targeted and general planner behavior.

## 2) Additional note on TPC-H/TPC-DS coverage

**Reviewer note**
> So the TPC-H/TPC-DS one is already a good one to optimize for.

**Proposed response**
Agreed. TPC-H/TPC-DS should remain the primary macro-level signal for optimization value and regression detection.

The intent here is to complement those suites with a deterministic micro-benchmark that isolates one known planner hotspot; macro benchmarks are still required to verify end-to-end relevance and prevent narrow wins.

**Plan**
1. Use this benchmark only as a focused diagnostic/perf guardrail for `PushDownFilter`.  
2. Continue validating any follow-up optimization with TPC-H/TPC-DS-style workloads before considering it complete.  
3. Include both micro and macro benchmark deltas in follow-up PR descriptions.
