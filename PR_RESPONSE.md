# PR Review Responses and Plan

## Context Reviewed
I reviewed the commit range `2fa93779d^..a2e36d585`, with focus on `.github/workflows/extended.yml` changes in the `sqllogictest-sqlite` job.

## Reviewer: @alamb

### Comment 1 (Build step)
> Rather than `cargo test` another idea would be to use `cargo build` 🤔  
> Also, what is the reason to use `--message-format=json`?  
> Finally, what is the `head` command for?

**Proposed response**
Thanks, good suggestions. I agree `cargo build` is clearer here since this step only compiles the `sqllogictests` target and does not run it.

I used `--message-format=json` to capture the exact emitted test binary path because Cargo places test executables under `target/.../deps` with a hash suffix. The `head -n 1` was a simple way to select the first extracted executable path from the stream.

I’ll simplify and document this more clearly in the workflow so the intent is explicit.

**Plan**
1. Replace the compile command with `cargo build --profile release-nonlto --features backtrace,parquet_encryption --package datafusion-sqllogictest --test sqllogictests`.
2. Add inline comments explaining why we capture the produced executable path (hashed filename under `target/.../deps`).
3. Tighten path extraction to target only `sqllogictests` output and avoid ambiguous parsing.
4. Keep the explicit empty-check guard and failure message if no executable is found.

---

### Comment 2 (Subshell and cd)
> Why the `(` and `)`? Also, why does it need to do `cd` when the current version doesn't?

**Proposed response**
Good callout. The `cd` is needed because `sqllogictests` resolves test data via relative paths (for example `test_files/` and `../../datafusion-testing/data/` in `datafusion/sqllogictest/bin/sqllogictests.rs`), so running from repo root can point it at the wrong locations.

The subshell `(...)` was only used to scope `cd` so it would not affect the subsequent `cargo clean`. It is not strictly necessary.

I’ll remove the subshell and switch to a cleaner step-level `working-directory: datafusion/sqllogictest` for the test execution step.

**Plan**
1. Set `working-directory: datafusion/sqllogictest` on the run step instead of using `cd` in script body. (done)
2. Remove `(` and `)` entirely. (done)
3. Add a short comment in the workflow noting that the test binary expects crate-relative paths for test files. (done)

---

## Summary of Intended Revision
- Use `cargo build` for compilation-only behavior.
- Clarify (with comments) why executable path capture is needed.
- Remove subshell/cd shell mechanics and use `working-directory` for readability.
- Preserve behavior and safeguards while making the workflow easier to understand and maintain.
