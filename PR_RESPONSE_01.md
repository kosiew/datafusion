# PR Review Response: Extended Tests Performance Investigation

**Reviewer**: @alamb  
**Date**: Review received 4 hours ago  
**Topic**: `.github/workflows/extended.yml` changes for sqllogictests timing

---

## Context

This PR aims to address the issue of extended tests taking over 2 hours to complete. The extended test performance regression was first identified after PR #19674 merged struct-field matching and validation logic. This change focuses on measurement and visibility rather than code optimization.

---

## Review Comment Summary

> I am very confused (similar to @nuno-faria) why this is taking so much less time. I would be hesitant to merge this until we understand why this would make the CI run faster.
>
> As an interim step, can you try at least splitting the build and run steps so we can see the timings?

The reviewer's key concerns:
1. **Unclear rationale**: Why would the current workflow changes make CI significantly faster?
2. **Lack of visibility**: Need separate timing measurements for build vs. test execution
3. **Cautious approach**: Split the steps explicitly before merging

---

## Response & Plan

### Addressing the Core Concern

**Acknowledgment**  
You're absolutely right to be cautious. I should clarify: **the current changes alone don't directly optimize test runtime.** 

The real value of this PR is:
1. **Establishing a measurement baseline** via separated build/test steps with timing visibility
2. **Profiling foundation** to identify which specific tests or logic became slower after PR #19674
3. **Enabling targeted optimization** once we have clear data on where time is spent

**The issue today**: We merged struct-matching changes that introduced overhead (new validation logic, expanded SQL test coverage), but we don't have clear visibility into *which* operations are causing the slowdown (compilation, test execution, or both). Splitting build and run is a prerequisite for root-cause analysis.

---

### Implementation Plan

I agree with your suggestion to split build and run explicitly. Here's the proposed revision:

#### **Step 1: Build Only**
```bash
cargo build \
  --profile release-nonlto \
  --features backtrace,parquet_encryption \
  --package datafusion-sqllogictest \
  --test sqllogictests
```

**Rationale**: 
- Builds only the test binary; no test execution.
- Isolates compilation time from test runtime.
- Timing of this step tells us if compilation regressed after PR #19674.

#### **Step 2: Extract Test Binary Path**
```bash
TEST_BIN=$(cargo build --profile release-nonlto --features backtrace,parquet_encryption --package datafusion-sqllogictest --test sqllogictests --message-format=json | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p' | head -n 1)
if [ -z "$TEST_BIN" ]; then
  echo "Could not find sqllogictests test binary"
  exit 1
fi
```

**Rationale**:
- Cargo places test executables under `target/.../deps` with hash suffixes; `--message-format=json` captures the exact path.
- Avoids brittle glob patterns.
- Explicit null check ensures early failure if something is wrong.

#### **Step 3: Run Tests**
```bash
cd datafusion/sqllogictest
"$TEST_BIN" --include-sqlite
```

**Rationale**:
- Runs the pre-built binary directly (no recompilation).
- Timing of this step isolates test execution from build overhead.
- `cd` into `datafusion/sqllogictest` because `sqllogictests` resolves test data via relative paths (e.g., `test_files/`, `../../datafusion-testing/data/`).

---

### Expected Outcome

Once these steps are in place and run in CI, we will have:

1. **Build timing**: How long did compilation take?
2. **Test timing**: How long did test execution take?

This gives us the **data needed to answer the real question**: Did PR #19674 slow down compilation, test runtime, or both?

---

### Next Steps (Post-Merge Profiling)

After this change is merged and the workflow runs:

1. **Collect baseline data** from the split timings.
2. **Compare against pre-PR #19674 runs** (if CI logs are available) or run profiling locally.
3. **Profile hot paths** identified in step 2 (e.g., struct validation, SQL test expansion).
4. **Apply targeted optimizations** such as:
   - Fast-path checks for identical schemas in `cast_struct_column()`
   - Lazy evaluation of compatibility checks
   - Inline hints for frequently-called validation functions
   - Early bailout in validation loops

---

## Revised Workflow Snippet

```yaml
- name: Build sqllogictest binary
  run: |
    # Compile test binary only; isolate build time for profiling.
    cargo build \
      --profile release-nonlto \
      --features backtrace,parquet_encryption \
      --package datafusion-sqllogictest \
      --test sqllogictests
    
    # Extract the test binary path (Cargo places executables under target/.../deps with hash suffix).
    TEST_BIN=$(cargo build --profile release-nonlto --features backtrace,parquet_encryption --package datafusion-sqllogictest --test sqllogictests --message-format=json | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p' | head -n 1)
    if [ -z "$TEST_BIN" ]; then
      echo "Error: Could not locate sqllogictests binary. Build may have failed."
      exit 1
    fi
    echo "TEST_BIN=$TEST_BIN" >> "$GITHUB_ENV"

- name: Run sqllogictest
  working-directory: datafusion/sqllogictest
  run: |
    # Run pre-built binary; isolate test runtime for profiling.
    # Test data resolved relative to datausion/sqllogictest directory.
    "$TEST_BIN" --include-sqlite
```

---

## Summary

| Aspect | Status | Rationale |
|--------|--------|-----------|
| **Split build/run** | ✅ Agreed | Enables separate timing measurement |
| **Error checking** | ✅ Included | Fails fast if binary not found |
| **Timing clarity** | ✅ Enabled | CI logs will show build time ≠ test time |
| **Next optimization** | ⏳ Pending | Requires profiling data from CI runs |

This approach acknowledges your concern and sets the foundation for data-driven optimization. Once we have CI timing data, we can confidently identify which PR #19674 changes caused overhead and apply targeted fixes.

---

**Questions for continuation**:
- Should we add explicit `echo` statements in the workflow to log timing delimiters (e.g., "Build started", "Test started")?
- Would it be helpful to add a follow-up task to track optimization work once profiling is complete?
