Rationale:

PushDownFilter is a known planning hotspot in sql_planner_extended, but we currently lack targeted observability and a focused benchmark shape to measure the suspected costly path (CASE/nullability inference under non-inner joins). This PR adds low-overhead, debug-gated timing/counter logs in PushDownFilter (simplify_predicates and infer_join_predicates) and introduces a reproducible CASE-heavy LEFT JOIN benchmark variant in sql_planner_extended.

Why this is needed:

Establishes actionable planner-time visibility without affecting non-debug performance.
Creates a stable benchmark to detect and prevent regressions in the exact hotspot shape reported by profiling.
Provides a baseline for follow-up optimization PRs (fast-path null-restrict checks and memoization) with measurable before/after impact.

---


Run these, in order:

cargo check -p datafusion-optimizer
cargo test -p datafusion-optimizer push_down_filter -- --nocapture
cargo check -p datafusion --bench sql_planner_extended
cargo bench -p datafusion --bench sql_planner_extended -- logical_plan_optimize_case_heavy_left_join --sample-size 10
cargo bench -p datafusion --bench sql_planner_extended -- logical_plan_optimize --sample-size 10


the direct goal is to make push_down_filter faster.

The A/B exists for a slightly different reason: to measure net planner impact of enabling that rule.

with_push_down_filter: real default behavior users get.
without_push_down_filter: control baseline if rule were absent.
Why that matters:

A rule can get faster internally but still be a net negative in some workloads.
Or it can be “slow” internally but still worthwhile if it enables better downstream plans.
A/B tells you whether the rule is helping or hurting end-to-end planning at each complexity point.
So both are needed:

Debug timing: optimize internals of push_down_filter.
A/B sweep: verify the rule is a net win (or at least not a regression) when enabled.