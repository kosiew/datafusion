source: pr-22529_a
# Centralize DataFusion SQL dialect metadata

## Summary

DataFusion currently keeps its own SQL dialect metadata in several separate places that must stay manually synchronized:

- `datafusion/common/src/config.rs`
  - `Dialect` enum variants
  - `Dialect::AVAILABLE`
  - `impl AsRef<str> for Dialect`
  - `impl FromStr for Dialect`

This creates drift risk whenever a dialect is added, renamed, aliased, or removed. Keep this refactor focused on DataFusion's configuration metadata; do not move SQL parser construction into `datafusion-common`.

## Problem

Adding a new dialect requires editing multiple independent lists/matches:

1. Add a `Dialect` enum variant.
2. Add the human-readable name to `Dialect::AVAILABLE`.
3. Add canonical serialization in `AsRef<str>` / `Display`.
4. Add config parsing and aliases in `FromStr`.
5. Update error messages and docs generated from config metadata.
6. Separately verify, in an SQL-enabled crate if needed, that canonical names remain accepted by `sqlparser::dialect::dialect_from_str`.

If any step is missed, users can see inconsistent behavior. Examples of possible drift:

- A dialect appears in an error message but is not accepted by config parsing.
- A dialect is accepted by `FromStr` but omitted from `Dialect::AVAILABLE`.
- Aliases are accepted by DataFusion config but not covered by tests.
- CLI/core error messages drift because they all rely on the same manually maintained available-dialect string.

The recent Spark dialect addition is correct, but it shows the maintenance cost: `Spark` needed updates in the enum, `AVAILABLE`, `AsRef`, `FromStr`, docs/config output, and downstream parser error messages.

## Proposed direction

Introduce one central dialect metadata source in `datafusion-common`, then derive the public behavior from it.

Possible shape:

```rust
struct DialectInfo {
    dialect: Dialect,
    canonical_name: &'static str,
    display_name: &'static str,
    aliases: &'static [&'static str],
}

const DIALECTS: &[DialectInfo] = &[
    DialectInfo {
        dialect: Dialect::Generic,
        canonical_name: "generic",
        display_name: "Generic",
        aliases: &[],
    },
    // ...
];
```

Then implement:

- `Dialect::as_ref()` from `canonical_name`
- `Display` from `canonical_name`
- `FromStr` from `canonical_name` + `aliases`
- `Dialect::AVAILABLE` or `Dialect::available()` from `display_name`

Keep parser construction outside `datafusion-common`. In particular, do not add a direct `sqlparser` dependency to `datafusion-common` for this refactor.

## Acceptance criteria

- Dialect names, aliases, display names, and canonical serialized values are defined in one place in `datafusion-common`.
- `Dialect::from_str`, `Dialect::as_ref`, and supported-dialect error messages are generated from that source.
- Existing accepted values continue to work, including aliases such as `postgres` and `sparksql`.
- Error messages remain clear and include all supported dialects.
- Existing config docs output remains equivalent except for intentional formatting changes.
- No new `sqlparser` dependency is added to `datafusion-common`.
- Tests cover:
  - every `Dialect` variant round-trips through `as_ref()` / `Display` / `FromStr`
  - documented aliases parse to the expected variant
  - available-dialect listing includes all variants exactly once

## Suggested tests

Add unit tests in `datafusion/common/src/config.rs`:

- Iterate over the central dialect table and assert canonical names parse back to the same variant.
- Assert every alias parses to the expected variant.
- Assert `Dialect::available()` or equivalent contains each display name once.

Optionally add an integration-level test in a crate that already depends on SQL parser support to ensure every canonical `Dialect::as_ref()` string accepted by DataFusion is also accepted by `sqlparser::dialect::dialect_from_str`. Do not add that dependency to `datafusion-common`.

## Notes

This is not a user-visible bug today. It is a maintainability refactor to reduce future regression risk when dialect support changes.
