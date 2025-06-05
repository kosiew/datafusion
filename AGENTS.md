# Repository Guidelines for Codex Agents

This repository uses Rust and contains documentation written in Markdown and TOML configuration files. When modifying code or docs, please follow these steps before committing:

1. **Run the Rust lints**
   ```bash
   ./dev/rust_lint.sh
   ```
   This script checks formatting (`cargo fmt`), lints (`cargo clippy`), validates `taplo` formatting and generates docs with warnings enabled.

2. **Format Markdown and TOML**
   ```bash
   # Format Markdown files
   prettier -w {datafusion,datafusion-cli,datafusion-examples,dev,docs}/**/*.md

   # Check TOML formatting
   taplo format --check
   ```

3. **Run the tests**
   ```bash
   cargo test
   ```
   You may also use `cargo nextest run` if installed.

4. **Optional: build documentation**
   ```bash
   ./docs/build.sh
   ```
   This renders the user and contributor guides.

Ensure these commands succeed before creating commits.
