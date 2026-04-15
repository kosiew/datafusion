source: pr-21944_a
# Centralize FFI `Result<SVec<SString>>` conversions

## Problem

`datafusion/ffi/src/catalog_provider.rs` and `datafusion/ffi/src/schema_provider.rs` contain near-identical bidirectional conversion code between DataFusion `Result<Vec<String>>` and the FFI ABI type `FFI_Result<SVec<SString>>`.

**Producer side** (DataFusion Rust → FFI ABI):
```rust
// catalog_provider.rs:107–113
unsafe extern "C" fn schema_names_fn_wrapper(
    provider: &FFI_CatalogProvider,
) -> FFI_Result<SVec<SString>> {
    unsafe {
        let names = sresult_return!(provider.inner().schema_names());
        FFI_Result::Ok(names.into_iter().map(|s| s.into()).collect())
    }
}
```

```rust
// schema_provider.rs:115–123
unsafe extern "C" fn table_names_fn_wrapper(
    provider: &FFI_SchemaProvider,
) -> FFI_Result<SVec<SString>> {
    unsafe {
        let provider = provider.inner();
        let table_names = sresult_return!(provider.table_names());
        FFI_Result::Ok(table_names.into_iter().map(|s| s.into()).collect())
    }
}
```

**Consumer side** (FFI ABI → DataFusion Rust):
```rust
// catalog_provider.rs:302–308
impl CatalogProvider for ForeignCatalogProvider {
    fn schema_names(&self) -> Result<Vec<String>> {
        unsafe {
            Ok(df_result!((self.0.schema_names)(&self.0))?
                .into_iter()
                .map(|s| s.into())
                .collect())
        }
    }
}
```

```rust
// schema_provider.rs:324–330
fn table_names(&self) -> Result<Vec<String>> {
    unsafe {
        Ok(df_result!((self.0.table_names)(&self.0))?
            .into_iter()
            .map(|s| s.into())
            .collect())
    }
}
```

## Impact

- **Boilerplate**: The same conversion idiom is repeated across multiple callsites and files, violating DRY.
- **Maintenance risk**: Future catalog/schema FFI methods (e.g., `table_names`, `table_exist`) can inadvertently diverge in error handling or conversion style.
- **Hidden invariant**: The fallible FFI string-list contract is implicit, scattered across wrappers and trait implementations, making it hard to audit or extend correctly.

## Solution

Introduce two helper functions in `datafusion/ffi/src/util.rs` to encode the conversion invariant once:

### Option A: Helper functions (preferred)
```rust
// datafusion/ffi/src/util.rs

/// Convert a Result<Vec<String>> from DataFusion to FFI ABI format.
///
/// Maps errors via sresult_return! and strings via stabby's Into.
pub fn result_vec_string_to_ffi(names: Result<Vec<String>>) -> FFI_Result<SVec<SString>> {
    unsafe {
        let names = sresult_return!(names);
        FFI_Result::Ok(names.into_iter().map(|s| s.into()).collect())
    }
}

/// Convert a FFI_Result<SVec<SString>> to DataFusion Result<Vec<String>>.
///
/// Maps errors via df_result! and strings via stabby's Into.
pub fn ffi_result_vec_string_to_result(ffi_result: FFI_Result<SVec<SString>>) -> Result<Vec<String>> {
    unsafe {
        Ok(df_result!(ffi_result)?
            .into_iter()
            .map(|s| s.into())
            .collect())
    }
}
```

### Call-site refactoring
Replace each wrapper:
```rust
// Before
FFI_Result::Ok(names.into_iter().map(|s| s.into()).collect())

// After
result_vec_string_to_ffi(Ok(names))
```

And each consumer impl:
```rust
// Before
Ok(df_result!((self.0.schema_names)(&self.0))?
    .into_iter()
    .map(|s| s.into())
    .collect())

// After
ffi_result_vec_string_to_result(df_result!((self.0.schema_names)(&self.0))?)
```

## Acceptance Criteria

- [ ] Two helper functions added to `datafusion/ffi/src/util.rs` with doc comments explaining the FFI contract.
- [ ] All calls to `schema_names`, `table_names` FFI wrappers refactored to use helpers (4+ sites across catalog_provider.rs and schema_provider.rs).
- [ ] Existing FFI round-trip tests continue to pass.
- [ ] No behavior changes to external FFI ABI or consumer callers.

## Effort

Small; a helper function or macro in `ffi/src/util.rs` plus call-site simplification (5–10 min changes, straightforward review).

## Related

- Introduced in PR #21944 (catalog/schema provider fallibility).
- Complements FFI error-handling contract uniformity.
