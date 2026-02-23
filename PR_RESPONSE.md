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
