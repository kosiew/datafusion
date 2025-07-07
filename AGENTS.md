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

## Rust Code Guidelines

When generating or modifying Rust code in this repository, follow these style and quality rules:

### Idiomatic Rust
- Use `snake_case` for variables, functions and modules.
- Use `CamelCase` for types, traits and enum variants.
- Prefer pattern matching and destructuring over indexing.
- Use iterators rather than explicit loops when applicable.
- Utilize strong typing and enums rather than raw values.
- Use `Option<T>` for optional values and `Result<T, E>` for fallible operations.
- Derive standard traits such as `Debug` and `Clone` where appropriate.

### Clippy Compliance
- Avoid unnecessary clones and allocations.
- Minimize `unwrap()` calls; prefer the `?` operator or proper error handling.
- Do not use `expect()` or `panic!()` except in tests; use proper error handling instead.
- Follow conventional lifetime names and avoid large stack allocations.
- Use explicit integer types and avoid wildcard imports.

### Error Handling
- Propagate errors with `Result` and the `?` operator.
- Use proper error handling with `Result` types
- Implement custom errors with the `thiserror` crate when needed.
- Add context when returning errors to aid debugging.

### Performance Considerations
- Prefer `&str` parameters over `String` when ownership isn't required.
- Use `Arc` for shared ownership and zero-copy techniques where possible.

### Code Organization
- Keep functions focused and under roughly 40–50 lines.
- Reuse existing utilities and extend them before adding new ones.
- Group related functionality into modules or `impl` blocks.

### Documentation and Comments
- Document public APIs with `///` comments and examples where helpful.
- Explain non-obvious logic with comments that describe "why" rather than "what".

Following these conventions helps maintain a consistent and high quality codebase.
