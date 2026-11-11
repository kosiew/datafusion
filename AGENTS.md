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
- [DataFusion 56 Upgrade Guide](docs/source/library-user-guide/upgrading/56.0.0.md)

## Before Starting Work

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

## Testing

When adding tests, you MUST follow
[Choosing What Kind of Test to Write](docs/source/contributor-guide/testing.md#choosing-what-kind-of-test-to-write):

If documentation files changed then run
```bash
./ci/scripts/doc_prettier_check.sh --write --allow-dirty
```

Otherwise, run the workspace and extended suites separately:
```bash
cargo xtask ci step test workspace
cargo xtask ci step test extended
cargo xtask ci step test fuzz
```

The extended suite runs only the feature-required `extended_tests` target;
ordinary tests belong in the workspace suite. See
[Workflow / CI](AGENTS_EXTRA.md#workflow--ci) for feature-gating rules.

For modified code identify local benchmarks(if any) and run them against `main`. See [Benchmarks](benchmarks/README.md).
For microbenchmark changes, run `cargo xtask ci step test microbenchmarks` to
execute Criterion cases once with debug checks, without measuring performance.
This full suite is local-only; no hosted CI job or required check runs it.
See [Workflow / CI](AGENTS_EXTRA.md#workflow--ci) for scope and CI scheduling.

## Agent Skills

Repository-specific agent skills live under `.ai/skills/`. Each subdirectory is
a single skill with a `SKILL.md` (YAML frontmatter + body). Check that
directory for applicable skills before working on a task; new skills go in
`.ai/skills/<skill-name>/SKILL.md`.
