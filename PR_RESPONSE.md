# PR Review Responses

## 1) Comment on CASE-heavy benchmark shape (`datafusion/core/benches/sql_planner_extended.rs`)

**Reviewer comment**
> I don't think we really want a benchmark for the case expression.  
> We want to optimize for the evaluation cost of the filter during pushdown, so perhaps it could be written not using a large case expression as is done currently or adaptive removing filters, etc.

**Proposed response**
The reviewer’s worry is that by using a huge CASE expression we might be tuning for an unrealistic “case expression” workload instead of the more common cost of pushing filters through joins. 

To address that concern: the benchmark only uses CASE because that form triggered a profiler hotspot in `PushDownFilter` — the nullability/type‑inference codepath for filters on non‑inner joins. I don’t believe real‑world queries typically look like this, so the presence of CASE is purely a convenient way to exercise that particular expensive planner path, not the target of optimization.

To make this clear and avoid overfitting, I’m going to treat the CASE variant as a narrowly scoped micro‑benchmark and add a companion LEFT JOIN query with a simple predicate instead of a CASE. With both in place we can separate:

1. the baseline cost of pushing a filter through a join, and  
2. the extra work incurred when a CASE expression forces nullability inference.

That way the benchmark remains useful for optimization while still reflecting more general planner behaviour.

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
