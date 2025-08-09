# Repository Guidelines for Codex Agents

This repository uses Rust and contains documentation in Markdown and TOML configuration files. Our focus is on delivering robust, maintainable, and high-quality solutions. Please follow these guidelines when contributing:

---

## 1. Solution-First Approach

- **Understand the Problem:** Before writing code, ensure you clearly understand the requirements and desired outcomes. Ask clarifying questions if anything is ambiguous.
- **Design Thoughtfully:** Sketch out the architecture or data flow. Consider performance, scalability, and readability.
- **Provide Examples:** Include usage examples or code snippets demonstrating how your feature or fix works.

## 2. Code Quality & Best Practices

### Idiomatic Rust

- Prefer `snake_case` for variables, functions, and modules; `CamelCase` for types and enums.
- Use pattern matching, `Option<T>`, and `Result<T, E>` for clear and safe handling of optionality and errors.
- Leverage iterators and functional constructs for concise, efficient code.

### Maintainability

- Keep functions focused (ideally under 40 lines) and modules cohesive.
- Name functions and variables descriptively; avoid generic names like `foo` or `temp`.
- Group related types and functions and document public APIs with `///` comments and examples.

### Error Handling & Context

- Use the `?` operator to propagate errors with context. Avoid silent failures.
- Implement custom errors via the `thiserror` crate when appropriate.
- Provide clear messages to aid debugging and user feedback.

### Performance Considerations

- Favor zero-copy patterns: use `&str` over `String` and `Arc<T>` for shared data.
- Avoid unnecessary heap allocations; minimize cloning.
- Benchmark critical paths when performance is a concern.

## 3. Testing & Validation

- **Unit Tests:** Cover individual functions and edge cases with `cargo test`.
- **Integration Tests:** Validate end-to-end behavior, especially for CLI commands or key modules.
- **Continuous Testing:** Ensure tests run reliably in CI. Use `cargo nextest` for parallel execution if configured.

## 4. Documentation & Examples

- Provide clear README updates for new features or changes.
- Include practical examples in code comments and the `examples/` directory.
- Update CLI help strings and guides to reflect enhancements.

## 5. Collaboration & Review

- **Pull Requests:** Provide a concise summary of changes, motivation, and how to test.
- **Code Reviews:** Offer constructive feedback focusing on clarity, correctness, and design.
- **Discussions:** Use issues to propose major changes or ask design questions.

## 6. Optional Tooling Checks

While linting and formatting are important, they should not overshadow solution quality. Please run formatting and linting **after** finalizing your code:

```bash
# Optional but recommended:
./dev/rust_lint.sh    # Formats, lints, and checks docs
./pre-commit.sh       # Runs clippy and fmt for staged Rust files
prettier -w <path/to/file.md>  # Formats Markdown
taplo format --check  # Validates TOML
```

Ensure these checks pass before merging, but prioritize delivering clear, well-designed code.
