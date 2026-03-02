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