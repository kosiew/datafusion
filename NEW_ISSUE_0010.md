source: pr-21710_a
# GitHub Issue Draft

## Title
Design policy for value-dependent Spark UDF return types (starting with ceil(expr, scale))

## Labels
spark, design, type-system, tech-debt

## Body
### Summary
Some Spark-compatible UDFs need return types that depend on argument values, not just argument data types. `ceil(expr, scale)` is the current example and has exposed planner/runtime type-contract risk.

This issue proposes selecting and documenting a design pattern for value-dependent return typing, then applying it to `ceil` as a pilot.

### Problem statement
Current return type contract is `return_type(&[DataType]) -> Result<DataType>`, which cannot observe literal argument values.

For `ceil(expr, scale)`, runtime behavior can branch on `scale` values while planning only sees types. This allows planned schema and produced runtime type to diverge.

### Reproduction scenarios to protect against
- `ceil(float_col, 0)` should not plan one type and produce another at runtime.
- `ceil(float_col, non_zero_literal)` should preserve the selected contract consistently.
- `ceil(decimal_col, literal_scale)` should not claim one decimal output type while returning another.

### Why this matters
- Type-contract mismatches can break downstream assumptions and query correctness.
- Bugs are difficult to detect without explicit type + value contract tests.
- Other Spark UDFs may eventually require similar behavior.

### Decision options
Option A: Stable runtime type per signature

- Keep existing return type API.
- Require runtime to produce a single stable output type for each signature.
- Avoid value-dependent runtime type switching.

Option B: Literal-aware planning for selected UDFs

- Add planner hook with access to expression arguments (including literals).
- Compute return type with value context when arguments are literals.
- Define fallback for non-literal arguments.

Option C: Hybrid policy

- Default to stable types.
- Allow explicit opt-in for literal-aware typing in exceptional UDFs.
- Require stronger tests and docs for opt-in paths.

### Requested decision
Pick one policy (A/B/C) and record it in contributor-facing guidance.

### Scope
In scope:

- Selecting and documenting policy for value-dependent Spark UDF typing.
- Implementing policy for `ceil(expr, scale)`.
- Adding regression tests that validate both values and output types.

Out of scope:

- Full redesign of all DataFusion UDF contracts in one change.
- Behavioral expansion unrelated to current `ceil` type-contract gap.

### Deliverables checklist
- [ ] ADR or equivalent design note for chosen policy.
- [ ] `ceil(expr, scale)` implementation aligned to policy.
- [ ] Contributor guidance update for future Spark UDFs.
- [ ] Test pattern added for value + output-type assertions.

### Acceptance criteria
- [ ] `ceil(expr, scale)` planned schema always matches runtime output type.
- [ ] Tests cover float with scale `0` and non-zero literal scales.
- [ ] Tests cover integer with negative scale.
- [ ] Tests cover decimal with explicit scale.
- [ ] Policy for literal vs non-literal scale arguments is explicit and tested.

### Testing plan
- Add unit tests for return-type contract behavior.
- Add UDF execution tests asserting values and concrete output `DataType`.
- Add at least one planner+execution test path that validates schema consistency end-to-end.

### Risks
- Literal-aware APIs can increase planner complexity.
- Underspecified policy can create inconsistent adoption across Spark UDFs.

### Open questions
1. Should value-dependent typing be allowed only when the dependent argument is a literal?
2. For non-literal scales, should we use a conservative supertype policy?
3. Should guidance live only under Spark docs or shared UDF contributor docs?
4. Are there backward-compat implications for existing Spark UDF output schemas?

### Related context
This issue is derived from PR review findings that identified a planner/runtime type contract mismatch risk in newly expanded 2-arg `ceil` behavior.
