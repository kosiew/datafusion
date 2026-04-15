created #21910
# Issue Proposal: Introduce a reusable execution-layer helper for schema normalization

## Summary

DataFusion execution plans expose a declared output schema, but some operators can still emit `RecordBatch` values whose physical batch schema drifts from that declared schema. PR #21770 fixes one concrete instance in `RecursiveQueryExec`, where recursive-term batches preserved their native field names instead of the anchor term's plan-declared names.

That PR appropriately uses `RecordBatch::try_new(...)` to rebuild the batch with the plan-declared schema, because this path needs to normalize field names and Arrow's `RecordBatch::with_schema(...)` enforces compatibility / containment checks that can reject that kind of rename.

This issue proposes introducing a small reusable execution-layer helper for this class of schema normalization so operators can express the intent clearly and handle similar cases consistently.

## Motivation

The recursive CTE bug shows a recurring execution-layer risk:

- the logical / plan-level schema is correct
- runtime batches coming from an independently planned branch carry different field names
- downstream consumers observe the batch-local schema rather than the plan contract

That matters for user-visible behavior because some downstream code keys directly on `batch.schema().field(i).name()`, including:

- sort / TopK-style paths that preserve incoming batches
- CSV / JSON writers
- custom collectors or adapters that inspect emitted batches directly

Right now, when an execution node needs to normalize emitted batches back to its declared output schema, the logic is local and ad hoc. That makes it harder to:

- spot where normalization is required
- document when the operation is a simple metadata rebind versus a schema rename
- audit similar operators in the future

## Why this should not just use `RecordBatch::with_schema(...)`

It is tempting to standardize on `RecordBatch::with_schema(...)`, but that API is not sufficient for all relevant cases.

In particular:

- `with_schema(...)` is ideal when the target schema is Arrow-compatible with the current batch under containment checks
- it is not ideal when the operator intentionally needs to normalize field names to the plan-declared schema
- the recursive CTE case in PR #21770 is exactly such a rename-driven normalization

That means any shared helper must be explicit about which normalization modes it supports. It should not accidentally hide the distinction between:

- compatible schema rebinding
- field-name normalization

## Problem Statement

We do not currently have a reusable execution-layer abstraction for:

- "this operator promises to emit batches matching its declared schema"
- "these child-produced batches need to be normalized before they leave this operator"
- "this normalization is allowed to rename fields, not just attach equivalent metadata"

As a result:

- operators implement the logic inline
- the rationale is easy to lose in future edits
- reviewers must re-derive the same invariants at each call site
- follow-up auditing for similar bugs is harder than it needs to be

## Proposed Direction

Introduce a small helper in `datafusion/physical-plan` or a nearby execution-focused module that makes schema normalization explicit.

Possible API shapes:

- `normalize_batch_schema(batch, expected_schema)`
- `rebind_batch_to_output_schema(batch, expected_schema)`
- a tiny helper type or utility module with focused functions

The implementation should be narrow and explicit. A reasonable first cut could support:

1. fast path: return the batch unchanged when the schema already matches
2. compatible rebind path: use an Arrow-backed check when simple rebinding is valid
3. rename normalization path: rebuild the batch with the declared schema when name normalization is intentionally required

If those paths need distinct semantics, it may be better to expose them as separate helpers rather than one overly-smart abstraction.

## Scope

In scope:

- define a reusable helper or helper pair for execution-layer schema normalization
- migrate `RecursiveQueryExec` to use the shared utility if that improves clarity
- document why name-normalizing call sites cannot always use `with_schema(...)`
- audit a small number of nearby operators that combine or forward batches from multiple branches

Out of scope:

- a large refactor of every `RecordBatch` construction site in the workspace
- changing Arrow's schema compatibility semantics
- redesigning logical planning or schema coercion rules

## Design Considerations

### 1. Keep the helper honest about semantics

The key design trap is over-generalization. A helper that silently mixes "equivalent schema rebind" and "intentional field rename normalization" could make the code less clear, not more.

The implementation should make the contract obvious at the call site.

### 2. Preserve zero-copy behavior where possible

When simple schema rebinding is valid, the helper should avoid data copies. But the API should not pretend every normalization path is identical if some cases require reconstructing the `RecordBatch`.

### 3. Make review easier

The value here is not just fewer lines of code. It is making the invariant visible:

- this operator may receive structurally-compatible batches
- before emission, those batches must match the operator's declared output schema

### 4. Avoid premature broadening

The payoff here appears moderate unless additional call sites emerge. That argues for a focused, low-complexity helper rather than a sweeping abstraction.

## Candidate Audit Targets

This issue is most relevant for operators that:

- merge results from independently planned branches
- replay or buffer child batches and emit them later
- forward batches without rebuilding them
- rely on plan-level coercion while preserving child-produced batch metadata

Examples worth auditing:

- recursive execution paths
- union-like or branch-combining execution nodes
- interleave / merge utilities
- serializer-facing boundaries where batch-local schemas matter

## Testing Plan

Add focused tests that cover:

- no-op behavior when the batch schema already matches the expected output schema
- successful normalization when field names need to be rewritten to the declared schema
- clear failure behavior when the requested normalization would violate column/type compatibility
- at least one execution-level regression demonstrating that emitted batches match the operator's declared schema

## Expected Benefits

- clearer execution-layer contracts around emitted batch schemas
- less ad hoc schema-rebind logic at individual call sites
- easier future auditing for similar bugs
- more durable documentation of when Arrow's `with_schema(...)` is and is not the right tool

## Acceptance Criteria

- a reusable schema-normalization helper or helper pair exists in the execution layer
- the helper documents the distinction between simple schema rebinding and field-name normalization
- `RecursiveQueryExec` either uses the helper or remains a clearly-documented intentional exception
- focused tests cover the supported normalization behavior

## Origin

Source:  #21770. 
The PR surfaced one concrete manifestation, but the broader question of how execution nodes should normalize emitted batch schemas is pre-existing.
