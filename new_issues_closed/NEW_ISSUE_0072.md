stale
source: pr-22584_a
# Reusable callback-only FFI context wrappers

## Summary

DataFusion FFI context wrappers should use a reusable callback-only pattern so consumers never need to dereference `private_data` allocated by another library.

The recent `FFI_PhysicalOptimizerContext` work shows the risk: the wrapper exposes stable callback fields for some context access, but also reconstructs/clones Rust internals by reading `private_data`. That only works in same-library tests. Across a real dynamic-library boundary, `private_data` is owned by the producer of that FFI object and must be treated as opaque by the other side.

## Problem

FFI wrappers in `datafusion/ffi` rely on a core invariant:

> `private_data` is only accessed by the library that created it. Foreign consumers must interact only through stable `extern "C"` function pointers.

Context-like traits are easy to get wrong because they are often short-lived borrowed adapters, not long-lived `Arc<dyn Trait>` wrappers. If each new context wrapper invents its own shape, it is likely to mix stable callbacks with direct Rust-side reconstruction of internals. That can cause:

- undefined behavior or crashes across dynamically loaded libraries
- silently invalid tests, because in-process unit tests share the same struct layout and allocation domain
- incomplete trait behavior when context methods are not represented as callbacks
- fragile future extensions when new context methods are added

## Concrete example

In PR #22584, `FFI_PhysicalOptimizerContext` needed to pass:

- `ConfigOptions`
- optional `StatisticsRegistry`

The implementation added callbacks for config and registry presence, but then read the registry via `context.inner().statistics_registry`. In the real FFI path, that context may have been created by the caller/consumer side. The rule-provider side cannot safely dereference that `private_data`.

This indicates a broader design gap: there is no shared, documented pattern for context trait wrappers that must remain opaque and callback-only.

## Proposed direction

Define a reusable context-wrapper pattern for `datafusion/ffi`:

1. Represent every exposed context method as an `unsafe extern "C" fn` field.
2. Treat `private_data` as fully opaque outside the creating library.
3. Do not reconstruct Rust internals from foreign `private_data`.
4. Use explicit FFI-safe wrappers for nested non-trivial objects, rather than cloning native Rust structs from foreign memory.
5. Add a small module-level template/comment for future context wrappers.
6. Require cross-library integration tests for every non-trivial callback.

For `PhysicalOptimizerContext`, possible outcomes:

- expose config-only context across FFI and document that `statistics_registry()` is unavailable, or
- define an FFI-safe `StatisticsRegistry`/statistics lookup callback surface and pass that through as opaque callbacks.

## Suggested implementation sketch

A context wrapper should look like:

```rust
#[repr(C)]
pub struct FFI_SomeContext {
    pub config_options: unsafe extern "C" fn(&Self) -> FFI_ConfigOptions,
    pub some_capability: unsafe extern "C" fn(&Self) -> FFI_Option<FFI_SomeValue>,
    pub release: unsafe extern "C" fn(&mut Self),
    private_data: *mut c_void,
}
```

Foreign-side use should call only callbacks:

```rust
let config = ConfigOptions::try_from(unsafe { (ctx.config_options)(ctx) })?;
let value = unsafe { (ctx.some_capability)(ctx) };
```

It should not do this:

```rust
ctx.inner().some_native_rust_field.clone()
```

## Test requirements

Add integration-test coverage under `datafusion/ffi/tests/` for context wrappers that cross dynamic-library boundaries:

- construct the context in one side of the FFI boundary
- call a foreign rule/provider/function that reads the context through callbacks
- verify the method works without local-bypass assumptions
- include one test that proves the non-default/context-aware path is used

For `PhysicalOptimizerContext`, an integration test should call `ForeignPhysicalOptimizerRule::optimize_with_context` and verify the rule sees the expected config/capability behavior.

## Benefits

- Preserves the FFI private-data ownership invariant.
- Prevents undefined behavior across dynamically loaded libraries.
- Makes future FFI context additions easier to audit.
- Forces end-to-end testing for ABI-sensitive callback surfaces.
- Reduces duplicated one-off wrapper logic.

## Scope

Medium.

This likely touches only `datafusion/ffi` initially, plus integration tests. Broader registry/statistics FFI support may require additional design if full `StatisticsRegistry` behavior must cross the boundary.
