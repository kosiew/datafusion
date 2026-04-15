source: pr-21710_a
# GitHub Issue Draft

## Title
Refactor Spark math UDFs: extract shared scale helpers for round and ceil

## Labels
spark, refactor, tech-debt

## Body
### Summary
`round` and `ceil` in Spark function math currently duplicate scale-parsing and scale-handling logic, and that duplication is already drifting.

This issue tracks extracting shared helpers so scale-aware Spark math UDFs use one consistent implementation path.

### Problem statement
The following drift is visible today:

- `ceil` has a `get_scale` helper that is effectively a copy of `round`, including `round`-specific error text.
- Numeric scale behavior is implemented separately per function, which increases the chance of subtle semantic divergence.
- Future scale-aware UDFs are likely to repeat this pattern if a shared helper layer is not introduced.

### Concrete evidence
- `ceil` scale parser emits `"round scale ..."` wording on i32-range failures.
- Both files have near-identical branches for signed/unsigned integer scale extraction and null-scale handling.

### Why this matters
- Increases maintenance burden for every bug fix in this area.
- Creates correctness risk from copy/paste drift.
- Makes Spark-compat behavior harder to audit because shared behavior is not centralized.

### Scope
In scope:

- Extracting shared scale argument parsing for Spark math UDFs.
- Centralizing utility math where behavior is intentionally common.
- Updating `round` and `ceil` to consume shared helpers.

Out of scope:

- Changing intended Spark semantics.
- Reworking generic DataFusion UDF APIs.
- Refactoring unrelated math functions.

### Proposed approach
Introduce a shared helper module, for example:

- `datafusion/spark/src/function/math/scale_utils.rs`

Candidate helper surface:

- `parse_optional_scale(function_name: &str, args: &[ColumnarValue]) -> Result<Option<i32>>`
- `scale_out_of_range_err(function_name: &str, value: impl Display) -> DataFusionError`
- shared `pow10` / factor helpers used by multiple scale-aware functions

Design constraints:

- Keep function-specific behavior explicit at call sites.
- Preserve null-scale semantics (`scale = NULL` propagates `NULL`).
- Keep diagnostics function-specific (`ceil ...` vs `round ...`).

### Task checklist
- [ ] Extract shared scale parser and wire `round` to it.
- [ ] Wire `ceil` to the shared parser and remove duplicate parser code.
- [ ] Remove `round`-specific error text from `ceil` paths.
- [ ] Consolidate only truly shared numeric helpers.
- [ ] Add unit tests for helper edge cases.
- [ ] Add one regression test proving `ceil` error text is `ceil`-specific.

### Acceptance criteria
- [ ] No duplicate scale parser remains across `round` and `ceil`.
- [ ] Invalid scale errors mention the correct function name.
- [ ] Existing valid-input behavior is unchanged unless explicitly fixed elsewhere.
- [ ] Shared helper module has focused unit tests:
- [ ] Signed/unsigned integer scale scalar coverage.
- [ ] NULL scale behavior.
- [ ] i32 conversion overflow and unsupported-type paths.

### Testing plan
- Run Spark math unit tests for `round` and `ceil`.
- Run new helper-module tests.
- Include at least one regression case for wrong-function error text.

### Risks
- Over-abstracting can reduce local readability.
- Refactor-only changes can accidentally alter semantics without edge-case tests.

### Related context
This issue comes from PR review feedback identifying duplication and drift between `ceil` and `round` scale logic.
