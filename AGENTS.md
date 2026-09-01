# Agent Guidelines for Apache DataFusion

Also read and follow [AGENTS_EXTRA.md](AGENTS_EXTRA.md).

## Developer Documentation

- [Quick Start Setup](docs/source/contributor-guide/development_environment.md#quick-start)
- [Testing Quick Start](docs/source/contributor-guide/testing.md#testing-quick-start)
- [Before Submitting a PR](docs/source/contributor-guide/index.md#before-submitting-a-pr)
- [Contributor Guide](docs/source/contributor-guide/index.md)
- [Architecture Guide](docs/source/contributor-guide/architecture.md)

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

When creating a PR, you MUST follow the [PR template](.github/pull_request_template.md):
describe the testing strategy and added/covering tests, explain no-test changes, and
check the Codecov reply covers changed code.

## Testing

If documentation files changed then run 
```bash
./ci/scripts/doc_prettier_check.sh --write --allow-dirty
```

Otherwise, run the default Rust CI test step:
```bash
cargo xtask ci step test workspace
```
Use `--explain` to print its underlying command.

For modified code identify local benchmarks(if any) and run them against `main`. See [Benchmarks](benchmarks/README.md).

## Agent Skills

Repository-specific agent skills live under `.ai/skills/`. Each subdirectory is
a single skill with a `SKILL.md` (YAML frontmatter + body). Check that
directory for applicable skills before working on a task; new skills go in
`.ai/skills/<skill-name>/SKILL.md`. Use
`audit-datafusion-spark-expression` when auditing a Spark-compatible function
against Spark.
