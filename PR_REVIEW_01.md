# PR Review
owner: apache
repo: datafusion
pr_number: issue-22034

## Decision
- [ ] Approve
- [x] Approve with suggestions
- [ ] Request changes

## Blocking findings
No blocking findings.

## Non-blocking suggestions
1. file: datafusion/physical-plan/src/common.rs
   line: 268
   side: RIGHT
   body: `SchemaAlignExec` conservatively rebuilds properties from a fresh `EquivalenceProperties`, which drops any child output ordering and hash partitioning even though the adapter preserves row order and values positionally. This is safe for correctness, but because `align_plan_to_schema` is a reusable public helper, consider preserving/remapping properties when schema-only changes allow it, or documenting the intentional loss to avoid unexpected optimizer/perf regressions as usage expands.

2. file: datafusion/physical-plan/src/common.rs
   line: 113
   side: RIGHT
   body: The core paths are covered, but most helper failure tests still call `project_plan_to_schema` rather than the new public `align_plan_to_schema`. Consider adding small align-specific tests for unchanged exact schemas, rename-only projection, and count/type/metadata/schema-metadata errors so the higher-level helper's selection and rejection behavior is locked down directly.

## High-impact refactor opportunities (out of scope)
No high-impact refactor opportunities.

## Follow-up actions
- Optional: add direct `align_plan_to_schema` unit coverage for the public helper contract.
- Optional: document or improve property preservation in `SchemaAlignExec`.

Focused validation run:
- `cargo test -p datafusion-physical-plan recursive_query_exec --quiet` passed
- `cargo test -p datafusion-physical-plan project_plan_to_schema --quiet` passed
- `cargo test -p datafusion-sqllogictest --test sqllogictests -- cte` passed
