source: pr-19420_a
# Feature Refactor: SessionConfig-driven preselection threshold for BinaryExpr

## Summary
`BinaryExpr::with_preselection_threshold` exists, but today the threshold is only configurable by directly constructing/modifying physical expressions. There is no standard user-facing path through `SessionConfig` or SQL planning flow.

This issue proposes introducing a session-level configuration and wiring it into physical planning/optimization so users can control binary preselection behavior without manual plan rewriting.

## Problem Statement
The preselection-threshold feature is currently difficult to use in real workloads:

- no `ConfigOptions` field
- no session-level knob
- no optimizer pass to apply threshold at planning time

As a result, users relying on SQL APIs cannot practically tune the threshold, despite the capability existing at expression level.

## Why This Matters
- Usability: feature is effectively inaccessible for most SQL/DataFrame users.
- Operability: no central runtime knob for workload tuning.
- Consistency: execution-policy behavior should be configurable through standard session config patterns.

## Goals
1. Add a user-facing session config key for binary preselection threshold.
2. Apply that threshold to relevant physical `BinaryExpr` nodes during planning/optimization.
3. Preserve default behavior when config is unset (or set to default value).
4. Ensure behavior is testable and documented.

## Non-Goals
- No SQL syntax hint design in this issue (session config only).
- No per-query hint parser changes.
- No broad rewrite of expression planning architecture.

## Proposed Design

### 1. Add ConfigOptions entry
Introduce a config option under execution namespace, for example:

- `datafusion.execution.binary_expr_preselection_threshold`

Expected semantics:
- Type: floating-point (`f32` or `f64` storage as appropriate)
- Default: `0.2` (current behavior)
- Valid range: `[0.0, 1.0]`, finite

Validation should reject invalid values with clear errors.

### 2. Apply config during physical planning/optimization
Add or extend a physical-optimizer rule that traverses physical expressions and sets `preselection_threshold` on `BinaryExpr` nodes.

Behavior:
- Apply to `BinaryExpr` nodes using `Operator::And` semantics (setting globally on all binary nodes is acceptable if harmless and simpler, but document rationale).
- Preserve explicit per-expression overrides if such overrides are introduced later.

### 3. Wire through SessionState/TaskContext
Ensure the optimizer/rule can access session config options from planning context and apply the configured threshold deterministically.

### 4. Document user-facing behavior
Update config docs and examples showing how to set and verify the option.

## Implementation Plan
1. Add config option definition and validation.
2. Add rule (or planner hook) to propagate config value onto `BinaryExpr` nodes.
3. Register rule in physical optimizer pipeline in correct order.
4. Add tests for default behavior and configured behavior.
5. Update docs and config references.

## Testing Plan

### Unit tests
- Config parsing/validation:
  - accepts `0.0`, `0.2`, `1.0`
  - rejects NaN, inf, negative, and >1.0 values

### Optimizer/planner tests
- Given a plan containing `AND` binary expressions and non-default config, assert resulting physical expressions have configured threshold.
- With default config, behavior remains unchanged.

### Execution tests
- End-to-end evaluation test demonstrating changed preselection path under different session config values.

### Optional proto test (if thresholds are serialized)
- Ensure configured threshold survives plan roundtrip serialization.

## API and UX Considerations
- Keep naming aligned with existing config conventions in DataFusion.
- Include clear doc text on threshold meaning:
  - `0.0` disables preselection
  - `1.0` always allows preselection for mixed LHS
- Mention that this is a performance/execution-policy knob and may affect error-surface timing in RHS evaluation.

## Risks and Mitigations
- Risk: unexpected behavior changes for existing workloads.
  - Mitigation: keep default at current value and document clearly.
- Risk: config applied inconsistently across planning paths.
  - Mitigation: centralize via one optimizer rule and add plan-level assertions.
- Risk: over-applying the setting to irrelevant nodes.
  - Mitigation: target only applicable binary operators or explicitly document all-node application.

## Acceptance Criteria
- New session config key exists with validation and docs.
- Physical planning/optimization applies configured threshold to relevant `BinaryExpr` nodes.
- Default behavior remains unchanged when option not set.
- Tests cover config validation, plan propagation, and execution impact.

## Suggested Scope Labeling
- `enhancement`
- `config`
- `physical-optimizer`
- `physical-expr`
- `docs`
