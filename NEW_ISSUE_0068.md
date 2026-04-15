source: pr-22529_a
# Centralize SQL dialect names, aliases, and parser mapping

## Summary

DataFusion currently keeps SQL dialect metadata in several separate places that must stay manually synchronized:

- `datafusion/common/src/config.rs`
  - `Dialect` enum variants
  - `Dialect::AVAILABLE`
  - `impl AsRef<str> for Dialect`
  - `impl FromStr for Dialect`
- Call sites that convert the configured DataFusion dialect into a `sqlparser` dialect via `sqlparser::dialect::dialect_from_str(...)`, for example in:
  - `datafusion/core/src/execution/session_state.rs`
  - `datafusion-cli/src/exec.rs`
  - `datafusion-cli/src/helper.rs`

This creates drift risk whenever a dialect is added, renamed, aliased, or removed.

## Problem

Adding a new dialect requires editing multiple independent lists/matches:

1. Add a `Dialect` enum variant.
2. Add the human-readable name to `Dialect::AVAILABLE`.
3. Add canonical serialization in `AsRef<str>` / `Display`.
4. Add config parsing and aliases in `FromStr`.
5. Ensure the canonical string is accepted by `sqlparser::dialect::dialect_from_str`.
6. Update error messages and docs generated from config metadata.

If any step is missed, users can see inconsistent behavior. Examples of possible drift:

- A dialect appears in an error message but is not accepted by config parsing.
- A dialect is accepted by `FromStr` but converts to a string that `sqlparser::dialect_from_str` does not support.
- Aliases are accepted by DataFusion config but not documented.
- CLI/core/sqllogictest error messages disagree about supported dialects.

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
- Optional helper for parser conversion/error validation, e.g. `Dialect::to_sqlparser_dialect()` behind the relevant feature boundary if acceptable

If avoiding a direct `sqlparser` dependency in `datafusion-common` remains important, keep parser construction outside `datafusion-common`, but expose a single canonical string and available-name list from the central metadata.

## Acceptance criteria

- Dialect names, aliases, display names, and canonical serialized values are defined in one place.
- `Dialect::from_str`, `Dialect::as_ref`, and supported-dialect error messages are generated from that source.
- Existing accepted values continue to work, including aliases such as `postgres` and `sparksql`.
- Error messages remain clear and include all supported dialects.
- Existing config docs output remains equivalent except for intentional formatting changes.
- Tests cover:
  - every `Dialect` variant round-trips through `as_ref()` / `Display` / `FromStr`
  - documented aliases parse to the expected variant
  - available-dialect listing includes all variants exactly once

## Suggested tests

Add unit tests in `datafusion/common/src/config.rs`:

- Iterate over the central dialect table and assert canonical names parse back to the same variant.
- Assert every alias parses to the expected variant.
- Assert `Dialect::available()` or equivalent contains each display name once.

Optionally add an integration-level test in a SQL-enabled crate to ensure every canonical `Dialect::as_ref()` string accepted by DataFusion is also accepted by `sqlparser::dialect::dialect_from_str`.

## Notes

This is not a user-visible bug today. It is a maintainability refactor to reduce future regression risk when dialect support changes.
