# PR Review Responses and De-scope Plan

## Context
Commit range reviewed: `92ef5e1bf^..b4cf47fe2`

I agree with the core review theme: this change set became too broad. The right next step is to split the work into minimal, independently reviewable PRs and keep this line of work focused on the core `CastColumnExpr` behavior.

---

## 1) `schema_rewriter.rs` (`index_of(column.name())` fix)

### Reviewer comment
> this change seems like it could be its own PR

### Response
Agreed. This is a correctness fix in `PhysicalExprAdapter` that is valuable on its own and should not be blocked by broader cast/proto discussions.

### Plan
- Extract into a standalone PR:
  - `datafusion/physical-expr-adapter/src/schema_rewriter.rs`
  - focused tests under adapter/parquet expr adapter coverage
- Scope: only column-name-based physical field lookup + test updates.
- Keep this PR independent so it can be cherry-picked/backported quickly.

---

## 2) `custom_file_casts.rs` nullable schema tweak

### Reviewer comment
> This could also be its own PR

### Response
Agreed. The example nullability adjustment is not required to review `CastColumnExpr` mechanics and should be separated.

### Plan
- Move to a small docs/examples PR:
  - `datafusion-examples/examples/custom_data_source/custom_file_casts.rs`
- Scope: example correctness/readability only.
- No coupling with proto or core cast expression internals.

---

## 3) `PhysicalCastColumnNode` in protobuf should be separate

### Reviewer comment
> This should be its own PR

### Response
Agreed. Adding new physical expression protobuf nodes significantly expands review surface (schema changes, generated code, serde paths) and should be independent.

### Plan
- Remove protobuf expansion from the core `CastColumnExpr` PR.
- Create dedicated proto PR(s):
  1. `PhysicalCastOptions` / cast options evolution for existing `PhysicalCastNode` (if still needed),
  2. optional later PR for `PhysicalCastColumnNode` only if we decide it must be serialized.
- Keep generated file updates (`prost.rs`, `pbjson.rs`) only in those proto-focused PRs.

---

## 4) `OwnedFormatOptions` duplicate type concern

### Reviewer comment
> Should we make changes to Arrow?

### Response
Good point. Long-term, upstreaming an Arrow-friendly ownership story is preferable to permanent duplicate local types.

### Plan
- Short-term: keep the owned wrapper only where required for DataFusion runtime-owned formatting/cast options.
- Medium-term: open/track an Arrow issue/proposal for reducing `'static` friction in `FormatOptions`/`CastOptions` usage.
- If upstream support lands, converge and remove/reduce local duplication.

---

## 5) Backward-compat fields in a new message

### Reviewer comment
> if this message is new, why do we need backwards compat fields?

### Response
Agreed; that comment is correct. A brand-new protobuf message does not need legacy compatibility fields at creation time.

### Plan
- Drop deprecated `safe` / `format_options` fields from `PhysicalCastColumnNode`.
- Keep compatibility logic only where evolving pre-existing messages (e.g., existing `PhysicalCastNode` semantics), not brand-new ones.

---

## 6) Concern about widening `CastColumnExpr` usage (proto exposure)

### Reviewer comment
> If our goal is to merge everything into `Cast` isn't this counterproductive?

### Response
Agreed. Exposing `CastColumnExpr` more broadly (especially via protobuf) conflicts with the stated long-term direction to converge on `CastExpr`.

### Plan
- Re-scope current work so `CastColumnExpr` remains an internal/targeted expression for schema-rewrite execution paths.
- Avoid public/proto expansion unless a clear long-term need is established.
- Follow-up design track: identify what `CastExpr` needs (field-level nullability/schema-aware struct behavior) so we can converge rather than proliferate expression variants.

---

## Proposed split sequence

1. **PR A (fast fix):** physical schema column lookup-by-name correctness + tests.
  branch: schema-lookup-fix-pr20202
2. **PR B (examples-only):** nullable schema correction in custom data source example.
  branch: custom-file-casts-nullable-pr20202
3. **PR C (core cast behavior):** `CastColumnExpr` implementation + validation/tests, no proto surface expansion.
  branch: castcolumnexpr-core-pr20202
4. **PR D (optional/proto):** cast proto evolution only after design alignment.
  branch: castcolumnexpr-proto-pr20202
5. **PR E (upstreaming):** Arrow discussion/proposal for ownership/lifetime ergonomics.

This sequence should reduce review latency and align each change with a single concern.

---

## Concrete branch/commit split plan (exact files + cherry-picks)

Below is a runnable split that preserves current work while minimizing scope per PR.

### Pre-step (once)

```bash
git fetch origin
git switch castcolumnexpr-20162a
```

### PR A — Physical schema lookup correctness only

**Branch:** `pr-a-schema-lookup-fix`

**Target files:**
- `datafusion/physical-expr-adapter/src/schema_rewriter.rs`
- `datafusion/core/tests/parquet/expr_adapter.rs`

**Suggested commits:**
- `64c339ca0`

**Commands:**

```bash
git switch -c pr-a-schema-lookup-fix origin/main
git cherry-pick 64c339ca0
```

---

### PR B — Example-only nullable schema tweak

**Branch:** `pr-b-custom-file-casts-nullable`

**Target files:**
- `datafusion-examples/examples/custom_data_source/custom_file_casts.rs`

**Suggested commits:**
- `4f478ba9c`

**Commands:**

```bash
git switch -c pr-b-custom-file-casts-nullable origin/main
git cherry-pick 4f478ba9c
```

---

### PR C — Core CastColumnExpr behavior (no protobuf)

**Branch:** `pr-c-castcolumnexpr-core`

**Target files (exact):**
- `datafusion/physical-expr/src/expressions/cast_column.rs`
- `datafusion/common/src/nested_struct.rs`
- `datafusion/common/src/format.rs`
- `datafusion/common/src/lib.rs`

**Suggested commit sources:**
- full commits: `df69e7ecd`, `53b93203c`, `a4142812c`, `9ca8b48e7`, `1c654f7ab`, `4acb703f2`, `bf93e46e9`, `91d8d5cec`
- partial from mixed commits:
  - `92ef5e1bf` (only the four files above)
  - `070aa7a1e` (only `cast_column.rs`)

**Commands:**

```bash
git switch -c pr-c-castcolumnexpr-core origin/main

# Mixed commit: keep only core files
git cherry-pick -n 92ef5e1bf
git restore --staged --worktree -- \
  datafusion/physical-expr-adapter/src/schema_rewriter.rs \
  datafusion/proto/proto/datafusion.proto \
  datafusion/proto/src/physical_plan/from_proto.rs \
  datafusion/proto/src/physical_plan/to_proto.rs
git commit -m "feat(physical-expr): add CastColumnExpr core validation/eval (no proto)"

# Core follow-up commits
git cherry-pick df69e7ecd 53b93203c a4142812c 9ca8b48e7 1c654f7ab 4acb703f2 bf93e46e9 91d8d5cec

# Mixed commit: keep only cast_column.rs test/code adjustments
git cherry-pick -n 070aa7a1e
git restore --staged --worktree -- datafusion/physical-expr-adapter/src/schema_rewriter.rs
git commit -m "test(cast): align CastColumnExpr construction in tests"
```

---

### PR D — Proto expansion (optional, only after design alignment)

**Branch:** `pr-d-castcolumnexpr-proto`

**Target files (exact):**
- `datafusion/proto/proto/datafusion.proto`
- `datafusion/proto/src/physical_plan/from_proto.rs`
- `datafusion/proto/src/physical_plan/to_proto.rs`
- `datafusion/proto/src/generated/prost.rs`
- `datafusion/proto/src/generated/pbjson.rs`

**Suggested commit sources:**
- full commits: `6e9c5b902`, `119ccc71f`, `4f97fb245`
- partial from mixed commit:
  - `92ef5e1bf` (only proto files above)

**Commands:**

```bash
git switch -c pr-d-castcolumnexpr-proto origin/main

# Mixed commit: keep only proto files
git cherry-pick -n 92ef5e1bf
git restore --staged --worktree -- \
  datafusion/common/src/format.rs \
  datafusion/common/src/lib.rs \
  datafusion/physical-expr-adapter/src/schema_rewriter.rs \
  datafusion/physical-expr/src/expressions/cast_column.rs
git commit -m "feat(proto): add CastColumnExpr proto nodes and serde wiring"

git cherry-pick 6e9c5b902 119ccc71f 4f97fb245
```

---

### PR E — Parquet nullability alignment (if kept separate)

**Branch:** `pr-e-parquet-nullability-alignment`

**Target files:**
- `datafusion/datasource-parquet/src/row_filter.rs`
- `datafusion/core/src/datasource/physical_plan/parquet.rs`

**Suggested commits:**
- `3a7e1fb64`
- `58bd20313`

**Commands:**

```bash
git switch -c pr-e-parquet-nullability-alignment origin/main
git cherry-pick 3a7e1fb64 58bd20313
```

---

## Commits to exclude from split PRs

- `58d429c78` (spill_pool refactor; later reverted)
- `b4cf47fe2` (revert spill_pool to main)
- `7bdac23d2` (`PR_REVIEW.md` only)

These should not be part of the functional split PRs above.
