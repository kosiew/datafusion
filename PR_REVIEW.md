# Pull Request Review: Benchmark Compilation Optimization

**Issue**: [#19072](https://github.com/apache/datafusion/issues/19072) - Benchmark binary compilation takes ~5 minutes  
**Commits Reviewed**: `b7026e2c6^..1fb62c651`  
**Date**: January 15, 2026  
**Reviewer**: GitHub Copilot

---

## Summary

This PR introduces feature flags to gate individual benchmarks and adds `CARGO_PROFILE` support to the benchmark script, aiming to reduce compilation time for the benchmarks binary. The changes enable developers to build only the benchmarks they need, significantly reducing build times and dependencies.

### Key Changes
1. **Feature-gated benchmarks** - 10 new features (`bench-*`) to selectively enable benchmarks
2. **Optional dependencies** - Made `serde`, `serde_json`, `parquet`, `object_store`, `rand`, and `tokio-util` optional
3. **CARGO_PROFILE support** - Added environment variable support in `bench.sh` script
4. **Disabled command handling** - Clear error messages when attempting to run disabled benchmarks
5. **Comprehensive test script** - Created `test_benchmark_changes.sh` for validation
6. **Documentation updates** - Updated README with feature flag usage examples

---

## Review Findings

### ✅ Strengths

1. **Effective Solution**: Directly addresses the compilation time issue by allowing selective compilation
2. **Well-structured Feature Design**: The `bench-common` pattern effectively shares base dependencies while keeping benchmark-specific ones isolated
3. **User-Friendly Error Messages**: The `disabled_benchmark()` function provides clear guidance on which feature to enable
4. **Comprehensive Testing**: The test script covers 10 different test scenarios with clear pass/fail indicators
5. **Excellent Documentation**: README examples are practical and cover common use cases
6. **Backward Compatible**: Default behavior (`bench-all`) maintains existing functionality

### 🔍 Issues & Suggestions

#### **Critical Issues** ❌

None identified. The implementation is functionally sound.

#### **Important Considerations** ⚠️

1. **Feature Duplication Pattern**

   **Location**: [benchmarks/src/bin/dfbench.rs](benchmarks/src/bin/dfbench.rs#L66-L118)
   
   The `Options` enum has significant duplication with paired `#[cfg(feature)]` / `#[cfg(not(feature))]` entries for each benchmark.
   
   **Current pattern** (repeated 10 times):
   ```rust
   #[cfg(feature = "bench-tpch")]
   Tpch(tpch::RunOpt),
   #[cfg(not(feature = "bench-tpch"))]
   #[command(name = "tpch")]
   Tpch(DisabledCommand),
   ```
   
   **Similar duplication in `main()`** (lines 131-173) with 20 match arms (2 per benchmark).
   
   **Suggestion**: Consider extracting into a macro to reduce boilerplate and maintenance burden:
   ```rust
   macro_rules! benchmark_option {
       ($variant:ident, $feature:literal, $module:path) => {
           #[cfg(feature = $feature)]
           $variant(<$module>::RunOpt),
           #[cfg(not(feature = $feature))]
           #[command(name = stringify!($variant).to_lowercase())]
           $variant(DisabledCommand),
       };
   }
   
   #[derive(Debug, Subcommand)]
   enum Options {
       benchmark_option!(Tpch, "bench-tpch", tpch),
       benchmark_option!(Tpcds, "bench-tpcds", tpcds),
       // ... etc
   }
   ```
   
   This would:
   - Reduce ~100 lines of repetitive code
   - Make adding new benchmarks easier
   - Reduce chance of copy-paste errors
   - Keep cfg logic centralized

2. **Test Script Location & Integration**

   **Location**: [test_benchmark_changes.sh](test_benchmark_changes.sh)
   
   The test script is at the repository root but specifically tests benchmarks functionality.
   
   **Considerations**:
   - Should this be in `benchmarks/test_feature_flags.sh`?
   - Should it be integrated into CI?
   - Is it meant to be temporary (commit-specific) or permanent?
   
   The script header says:
   ```bash
   # Test script for benchmark feature flags and CARGO_PROFILE changes
   # Commits: b7026e2c6^..f67eb945f
   ```
   
   **Suggestion**: If keeping the script permanently:
   - Remove commit references from header
   - Move to `benchmarks/` directory or `dev/` directory
   - Add to CI workflow to prevent regressions
   - Or if temporary, remove before merging

3. **Documentation of Build Time Improvements**

   **Location**: [benchmarks/README.md](benchmarks/README.md)
   
   The README explains *how* to use feature flags but doesn't quantify the build time improvements.
   
   **Suggestion**: Add a section showing actual build time comparisons:
   ```markdown
   ### Build Time Improvements
   
   Using selective features can significantly reduce build times:
   
   | Configuration | Build Time | Dependencies |
   |---------------|------------|--------------|
   | `--features bench-all` | ~5 minutes | 500+ crates |
   | `--features bench-tpch` | ~2 minutes | 300+ crates |
   | `--no-default-features --features bench-tpch` | ~1.5 minutes | 250+ crates |
   
   *Times measured on [hardware spec] with clean build*
   ```
   
   This would help users understand the actual impact and justify the feature flag approach.

#### **Minor Improvements** 💡

4. **Inconsistent Box::pin Usage**

   **Location**: [benchmarks/src/bin/dfbench.rs](benchmarks/src/bin/dfbench.rs#L151-L173)
   
   Only `Imdb`, `Tpch`, and `Tpcds` use `Box::pin()`:
   ```rust
   Options::Imdb(opt) => Box::pin(opt.run()).await,
   Options::Tpch(opt) => Box::pin(opt.run()).await,
   Options::Tpcds(opt) => Box::pin(opt.run()).await,
   ```
   
   **Questions**:
   - Why only these three? Is it due to recursive async calls or large future sizes?
   - Should this be documented in the code?
   
   **Suggestion**: Add a comment explaining the rationale:
   ```rust
   // Box::pin required for IMDB, TPCH, and TPCDS due to large future sizes
   // from recursive query execution. See [issue/PR link]
   Options::Imdb(opt) => Box::pin(opt.run()).await,
   ```

5. **Feature Flag Naming Consistency**

   **Location**: [benchmarks/Cargo.toml](benchmarks/Cargo.toml#L40-L61)
   
   Most benchmarks follow `bench-{name}` pattern, but consider if `bench-hj` should be `bench-hash-join` for clarity.
   
   Current names are concise, which is good. Consider documenting the full names in README:
   ```markdown
   - `bench-hj` - Hash Join benchmark
   - `bench-smj` - Sort Merge Join benchmark
   - `bench-nlj` - Nested Loop Join benchmark
   ```

6. **CARGO_PROFILE Default Documentation**

   **Location**: [benchmarks/bench.sh](benchmarks/bench.sh#L43-L46)
   
   Good addition! The script now supports:
   ```bash
   CARGO_PROFILE=${CARGO_PROFILE:-release}
   if [[ -z "${CARGO_COMMAND}" ]]; then
       CARGO_COMMAND="cargo run --profile ${CARGO_PROFILE}"
   fi
   ```
   
   **Suggestion**: Add an example to the README showing the speed-vs-performance tradeoff:
   ```markdown
   ### Quick Development Builds
   
   For faster iteration during development:
   ```bash
   CARGO_PROFILE=dev ./bench.sh run tpch  # fastest build, slower runtime
   CARGO_PROFILE=release-nonlto ./bench.sh run tpch  # balanced
   CARGO_PROFILE=release ./bench.sh run tpch  # slowest build, fastest runtime
   ```

7. **DisabledCommand Type**

   **Location**: [benchmarks/src/bin/dfbench.rs](benchmarks/src/bin/dfbench.rs#L56-L57)
   
   ```rust
   #[derive(Debug, clap::Args)]
   struct DisabledCommand;
   ```
   
   This is clever - using an empty struct as a placeholder. Consider adding a doc comment:
   ```rust
   /// Placeholder type for disabled benchmark subcommands.
   /// When a benchmark feature is disabled, clap still needs a type for the variant,
   /// but it will never be instantiated since the disabled_benchmark() function
   /// returns early with an error.
   #[derive(Debug, clap::Args)]
   struct DisabledCommand;
   ```

8. **Test Script Output Quality**

   **Location**: [test_benchmark_changes.sh](test_benchmark_changes.sh)
   
   The test script has excellent output formatting with colors and clear pass/fail indicators. Good work!
   
   **Minor suggestion**: Consider adding a `--quiet` flag to suppress verbose output for CI environments:
   ```bash
   QUIET=${QUIET:-false}
   
   info() {
       if [ "$QUIET" = "false" ]; then
           echo -e "${YELLOW}➜${NC} $1"
       fi
   }
   ```

---

## Review Checklist Assessment

| Criterion | Status | Notes |
|-----------|--------|-------|
| **Consistency** | ✅ Pass | Follows existing patterns; cfg-gating is idiomatic |
| **Simplicity** | ⚠️ Good | Could reduce duplication with macros (non-blocking) |
| **Design** | ✅ Pass | Feature flag design is well-structured and appropriate |
| **Effectiveness** | ✅ Pass | Fully addresses the compilation time issue |
| **Scope** | ✅ Pass | Changes are focused and minimal |
| **Docs** | ✅ Pass | README is comprehensive with excellent examples |
| **Tests** | ✅ Pass | Comprehensive test script covers all scenarios |

---

## Decision: **✅ Approve with Suggestions**

### Rationale

This PR effectively solves the benchmark compilation time problem with a clean, backward-compatible design. The implementation is functionally complete, well-tested, and properly documented.

**Why Approve**:
- ✅ Solves the stated problem (5-minute build times)
- ✅ Clean, idiomatic Rust with proper cfg-gating
- ✅ Comprehensive testing and documentation
- ✅ No breaking changes (default behavior unchanged)
- ✅ User-friendly error messages for disabled benchmarks
- ✅ No logic errors or security issues

**Suggestions are Non-Blocking**:
The suggestions above (macro extraction, test script location, documentation enhancements) are quality-of-life improvements that can be addressed in follow-up PRs or as part of this PR at the author's discretion. None are blockers for merge.

---

## Recommended Next Steps

1. **Consider macro extraction** for the duplicated cfg patterns (reduces ~100 lines)
2. **Decide on test script permanence** - integrate into CI or remove before merge
3. **Add build time metrics** to README showing actual improvements
4. **Document Box::pin usage** for the three special-cased benchmarks

---

## Additional Comments

### On the Solution Approach

The feature flag approach is the right solution for this problem. Alternative approaches (separate binaries, workspace members) would be more complex and harder to maintain. The current design:
- Keeps the single binary model
- Allows fine-grained control
- Maintains backward compatibility
- Follows Rust ecosystem best practices

### On Code Quality

The code demonstrates good engineering practices:
- Clear separation of concerns
- Helpful error messages
- Comprehensive testing
- Practical documentation with examples

The duplication in `dfbench.rs` is the only notable code smell, but it's a minor concern given cfg-gating constraints in Rust.

---

## References

- Original Issue: [#19072](https://github.com/apache/datafusion/issues/19072)
- Related PR: [#18985](https://github.com/apache/datafusion/pull/18985) (where the issue was raised)
- Rust Feature Flag Best Practices: [Cargo Book - Features](https://doc.rust-lang.org/cargo/reference/features.html)

---

**Final Recommendation**: **Approve and merge**. This is a solid improvement to developer experience with no functional risks.
