# Codex Agent Rules

Repo = Apache DataFusion. Big Rust workspace. Docs/config = Markdown/TOML. Keep file short. Detailed task rules live in `~/.pi/agent/skills`.

## Always-on

- Code works. Tests prove.
- Solve right problem first. No speculative opt. No scope creep.
- Errors predictable. Messages actionable.
- Prefer simple, maintainable, documented design.
- Behavior change -> regression test.
- Behavior/docs/help change -> update docs/examples/help.

## Skill routing

Load when task matches:

- `datafusion-development`: impl/refactor/API/workspace conventions.
- `datafusion-testing`: unit tests, SQLLogicTests, feature builds, lint/docs checks.
- `datafusion-performance`: benchmarks, profiling, perf regressions, memory pressure.
- `datafusion-area-contracts`: functions, optimizer, execution, datasource, Substrait, session extensions, FFI, proto contracts.
- `datafusion-pr-review`: PR/diff review in repo.
- `schema-contract-review`: schema, field, projection, RecordBatch, scan, FFI/proto, serde, metadata boundaries.
- `test-driven-development` / `tdd`: feature/bugfix impl.
- `systematic-debugging` / `diagnose`: bugs, failing tests, regressions.

## Toolchain + quick cmds

- Rust toolchain pinned in `rust-toolchain.toml`; use repo toolchain.
- Start crate-scoped:

```bash
cargo build -p <crate>
cargo test -p <crate>
```

- Cross-crate change -> broader checks.
- Python docs/dev/benchmarks use `uv`; run `uv sync` when needed.
- Major Rust change -> after code stable:

```bash
./dev/rust_lint.sh
```

## Workspace map

- Core planning/runtime: `datafusion/core`, `datafusion/execution`, `datafusion/sql`.
- Optimizers/plans: `datafusion/optimizer`, `datafusion/physical-optimizer`, `datafusion/pruning`, `datafusion/physical-plan`.
- Shared: `datafusion/common`, `datafusion/common-runtime`, `datafusion/macros`.
- Catalog/datasources: `datafusion/catalog*`, `datafusion/datasource*`.
- Expr/functions: `datafusion/physical-expr*`, `datafusion/functions-*`, `datafusion/functions`.
- Serde/protocols: `datafusion/proto*`, `datafusion/substrait`.
- Binaries/tests: `datafusion-cli`, `datafusion-examples`, `benchmarks`, `test-utils`.

## Testing summary

- Planner/executor/expression behavior change -> unit tests + SLT when SQL-visible.
- Prefer SLT in `datafusion/sqllogictest/test_files/`. Avoid snapshots.
- SLT `SET` -> restore with `RESET <config>` before file end.
- Serialization/boundary contracts -> roundtrip/contract tests.

## Release branches

Backport/release work -> follow `docs/source/contributor-guide/release_management.md`: land on `main` first when possible, cherry-pick to `branch-*`, forward-port release-only fixes/changelog.
