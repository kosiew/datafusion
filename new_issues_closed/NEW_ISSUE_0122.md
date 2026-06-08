source: pr-23861_a
# Centralize schema conformance for union-like physical streams

## Problem
`UnionExec` and `InterleaveExec` can expose a union-derived schema while forwarding child `RecordBatch`es produced with each child's own schema.

A recent fix wrapped `UnionExec` child streams with a schema-conforming stream when a child stream schema differs from the union's declared schema. That addresses the direct `UnionExec::execute` path, but the same invariant is owned by both union-like stream combiners:

- `UnionExec` declares a schema from `union_schema(inputs)`.
- `InterleaveExec` also declares a schema from `union_schema(inputs)`.
- `UnionExec` emits one selected child stream per output partition.
- `InterleaveExec` combines child streams through `CombinedRecordBatchStream`.
- `CombinedRecordBatchStream::poll_next` currently yields each child `RecordBatch` unchanged.

This leaves the declared-schema == emitted-batch-schema rule dependent on each operator remembering to wrap or re-stamp its children locally.

## Why it matters
Downstream consumers often require every batch in a stream to match the stream schema exactly. This includes consumers using Arrow C Stream / FFI paths, such as Python `pyarrow.Table.from_batches` through DataFusion bindings.

If a union-like plan declares a nullable field because one input is nullable, but another child emits batches where the same field is `NOT NULL`, the data values are valid but the batch schema differs from the stream schema. Consumers that check schema equality reject the result.

Keeping schema conformance as ad hoc logic in individual operators also creates a maintainability risk: future union/interleave changes can preserve correct data while silently reintroducing schema drift.

## Invariant / desired behavior
For every physical operator that declares a schema derived from multiple child schemas, every emitted `RecordBatch` must have a schema exactly equal to the operator's declared schema.

For union-like operators specifically:

- `UnionExec::schema()` must equal every batch schema yielded by `UnionExec::execute`.
- `InterleaveExec::schema()` must equal every batch schema yielded by `InterleaveExec::execute`.
- `CombinedRecordBatchStream::schema()` must equal every batch schema yielded by `CombinedRecordBatchStream` when it is used as the stream boundary for a declared combined schema.
- Nullability, field names, field metadata, schema metadata, and extension metadata must be preserved according to the declared output schema.
- Re-stamping must not change array data, row count, ordering, partitioning semantics, metrics, cancellation behavior, or error propagation.

## Proposed direction
Move schema conformance to the shared stream-combiner boundary used by union-like operators, rather than relying on one-off local guards.

A minimal direction:

1. Keep a small helper/wrapper that re-stamps a `RecordBatch` with a target `SchemaRef` using `RecordBatch::try_new_with_options` and the original row count.
2. Reuse that helper for both `UnionExec` and `InterleaveExec`, or make `CombinedRecordBatchStream` conform each yielded batch to its own declared schema before returning it.
3. Avoid wrapping/re-stamping when the child stream schema already equals the target schema.
4. Add tests that prove both direct `UnionExec` and optimizer-created/direct `InterleaveExec` paths emit batches whose schemas exactly match their plan/stream schema.

Prefer one narrow helper with a clear contract over duplicating comments and wrapping logic in each operator.

## Scope
### In
- Identify the canonical schema-conformance boundary for union-like physical streams in `datafusion/physical-plan/src/union.rs`.
- Refactor existing `UnionExec` schema-restamping logic into that boundary or a shared helper.
- Ensure `InterleaveExec` cannot emit child batches whose schemas differ from its declared schema.
- Preserve current `UnionExec` behavior for mixed-nullability `UNION ALL` inputs.
- Add regression coverage for an `InterleaveExec` path with mixed child nullability.
- Keep error propagation from invalid re-stamping as a normal `DataFusionError` result.

### Out
- Changing logical union coercion rules.
- Changing type coercion or cast insertion behavior.
- Broad rewrites of `UnionExec`, `InterleaveExec`, partitioning, or distribution enforcement.
- Adding schema conformance to unrelated operators unless they share this exact union-like stream boundary.
- Relaxing schema equality requirements for downstream consumers.

## Acceptance criteria
- [ ] `UnionExec` no longer owns a unique copy of schema-conformance logic that `InterleaveExec` can bypass.
- [ ] `InterleaveExec::execute` cannot yield a `RecordBatch` whose schema differs from `InterleaveExec::schema()` for compatible child schemas.
- [ ] `CombinedRecordBatchStream`, or the streams passed into it, enforce the declared output schema consistently.
- [ ] Existing mixed-nullability `UNION ALL` regression tests still pass.
- [ ] A new regression test covers mixed-nullability children through `InterleaveExec` and asserts every emitted batch schema exactly equals the plan/stream schema.
- [ ] The implementation preserves row counts and columns when re-stamping batches.
- [ ] The implementation avoids unnecessary re-stamping when schemas already match.

## Tests / verification
- Unit test in `datafusion/physical-plan/src/union.rs` or an integration test that constructs `InterleaveExec` directly with children that have the same field types but different nullability, then collects output and asserts every `RecordBatch::schema()` equals `InterleaveExec::schema()`.
- Regression test that exercises the optimizer path where `ensure_distribution` rewrites a `UnionExec` into `InterleaveExec`, then verifies batch schemas match the physical plan schema.
- Existing `datafusion/core/tests/sql/union_nullable.rs` tests for direct `UNION ALL` mixed-nullability behavior.
- Run targeted Rust tests for the modified physical-plan/core test modules.

## Notes / open questions
- If `CombinedRecordBatchStream` is only intended for `InterleaveExec`, putting conformance inside it is likely the smallest invariant owner.
- If `CombinedRecordBatchStream` may later be reused where child schemas are expected to remain distinct, prefer an explicit constructor or wrapper name that states it conforms batches to the declared schema.
