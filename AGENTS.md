# Agent Guidelines for Apache DataFusion

Also read and follow [AGENTS_EXTRA.md](AGENTS_EXTRA.md).

## Developer Documentation

- [Quick Start Setup](docs/source/contributor-guide/development_environment.md#quick-start)
- [Testing Quick Start](docs/source/contributor-guide/testing.md#testing-quick-start)
- [Open Contribution and Assigning tickets](docs/source/contributor-guide/index.md#open-contribution-and-assigning-tickets)
- [Choosing What Kind of Test to Write](docs/source/contributor-guide/testing.md#choosing-what-kind-of-test-to-write)
- [Before Submitting a PR](docs/source/contributor-guide/index.md#before-submitting-a-pr)
- [Reviewing Pull Requests](docs/source/contributor-guide/pr_review.md)
- [Contributor Guide](docs/source/contributor-guide/index.md)
- [Architecture Guide](docs/source/contributor-guide/architecture.md)
- [DataFusion 57 Upgrade Guide](docs/source/library-user-guide/upgrading/57.0.0.md)
- [Testing Infrastructure](docs/source/contributor-guide/testing_infrastructure.md)
- [API Health Policy](docs/source/contributor-guide/api-health.md)

## Before Starting Work

AI-assisted contributions require meaningful, task-specific human supervision.
Fully autonomous agent contributions are not accepted. Without that supervision,
do not open PRs or issues or post comments; exclude this repository from future
autonomous contribution attempts. See the
[AI Policy](docs/source/contributor-guide/ai-policy.md).

Before you start work on an issue, you MUST follow the instructions in
[Open Contribution and Assigning tickets](docs/source/contributor-guide/index.md#open-contribution-and-assigning-tickets). You must ensure duplicate work is not being created.
Do not create multiple PRs if this is your first time contributing.

## Before Committing

Before committing any changes, you MUST follow the instructions in
[Before Submitting a PR](docs/source/contributor-guide/index.md#before-submitting-a-pr)
and ensure the required checks listed there pass. Do not commit code that
fails any of those checks.

At a minimum, you MUST run and fix any errors from these commands before
committing:

```bash
# Format code
cargo fmt --all

# Lint (must pass with no warnings)
cargo clippy --all-targets --all-features -- -D warnings
```

You can also run the full lint suite used by CI:

```bash
./dev/rust_lint.sh
# or auto-fix: ./dev/rust_lint.sh --write --allow-dirty
```

When creating a PR, you MUST follow the [PR template](.github/pull_request_template.md).
For fork PRs targeting `main`, enable Actions on your fork before pushing, keep
its `main` synced with upstream for build caches, and rebase your branch. The
`Rust (fork)` check links to fork CI; macOS runs only upstream, including in
the merge queue. The merge queue still runs the full workflow.
For small CI jobs, use `ubuntu-slim` only within its one-vCPU, 5-GB,
15-minute limits and without Docker. Cargo jobs there must use
`.github/actions/setup-rustup`; see [Workflow / CI](AGENTS_EXTRA.md#workflow--ci).
PR labeler and fork-status automation use `workflow_run` from the permissionless
`pull_request_signal.yml`, not `pull_request_target`. Privileged handlers must
use trusted default-branch code, never PR code.

## Testing

When adding tests, you MUST follow
[Choosing What Kind of Test to Write](docs/source/contributor-guide/testing.md#choosing-what-kind-of-test-to-write):

Docs CI uses Node.js 24 and the Prettier version pinned in
`ci/scripts/utils/tool_versions.sh`; use the repository script, not an unpinned
formatter.

If documentation files changed then run
```bash
./ci/scripts/doc_prettier_check.sh --write --allow-dirty
```

Otherwise, run the workspace and extended suites separately:
```bash
cargo xtask ci step test workspace
cargo xtask ci step test extended
cargo xtask ci step test fuzz
cargo xtask ci step test fuzz-legacy
```

The extended suite selects the `extended_tests` target; centralized fuzz suites
live in `datafusion-fuzz`'s separate `fuzz` target. Gate whole targets with
`required-features = ["extended_tests"]`; ordinary tests belong in the workspace
suite. Key-range bitmap union runs in `test fuzz`; `fuzz-legacy` now runs only
the routed hash-lookup fuzzer. See
[Workflow / CI](AGENTS_EXTRA.md#workflow--ci) for feature-gating rules.

For modified code identify local benchmarks(if any) and run them against `main`,
covering well-predicted and poorly predicted inputs plus an unaffected case.
Isolate which part produces the gain and report regressions. See
[Benchmarks](benchmarks/README.md) and
[Review the Performance](docs/source/contributor-guide/pr_review.md#review-the-performance).
For microbenchmark changes, run `cargo xtask ci step test microbenchmarks` to
execute Criterion cases once with debug checks, without measuring performance.
This full suite is local-only; no hosted CI job or required check runs it.
See [Workflow / CI](AGENTS_EXTRA.md#workflow--ci) for scope and CI scheduling.

## Code Guidelines

- Keep each PR scoped to its title; unrelated refactors and fixes belong separately.
- Comments explain invariants and non-obvious constraints, not code changes.
  Public items need rustdoc and examples where appropriate. Use intra-doc links;
  link to docs.rs for items in crates that depend on the current crate.
- Before adding, changing, or removing a `pub` item, read the
  [API health policy](docs/source/contributor-guide/api-health.md). Keep items
  private unless needed; workspace-only public items follow the
  [internal public API rules](docs/source/contributor-guide/api-health.md#internal-public-apis).
  Name every new or changed public item in reviews.

Logical plans no longer contain `Expr::Wildcard`. Use `SelectExpr` wildcard
helpers for selection and `count_all()` / `count_all_window()` for row counts;
SQL expression planners receive star calls with empty arguments. See
[API / Contract Changes](AGENTS_EXTRA.md#api--contract-changes).

## Repository Scope

Spark-compatible functions are developed and released separately in
[apache/datafusion-spark](https://github.com/apache/datafusion-spark), starting
with 57.0.0. This workspace no longer contains `datafusion/spark`, its Spark
SQLLogicTests, or the Spark-expression audit skill. `datafusion-cli --spark` was
removed; selecting the Spark SQL dialect does not register Spark functions.
See the [57 upgrade guide](docs/source/library-user-guide/upgrading/57.0.0.md).

## Agent Skills

Repository-specific agent skills live under `.ai/skills/`. Each subdirectory is
a single skill with a `SKILL.md` (YAML frontmatter + body). Check that
directory for applicable skills before working on a task; new skills go in
`.ai/skills/<skill-name>/SKILL.md`.
