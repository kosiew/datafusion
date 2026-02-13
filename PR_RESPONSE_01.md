# PR Review Response: Extended Tests Performance Investigation

**Reviewer**: @alamb  
**Date**: Review received 4 hours ago  
**Topic**: `.github/workflows/extended.yml` changes for sqllogictests timing

---

## Context

This PR aims to address the issue of extended tests taking over 2 hours to complete. The performance regression was first identified after PR #19674 merged struct-field matching and validation logic. This specific PR changes only the workflow file (`.github/workflows/extended.yml`) to separate build and test execution steps, with early results suggesting improved performance—though the mechanism is not yet fully understood.

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

### Addressing the Core Concern - Honest Assessment

**You're absolutely right to be confused.** After reviewing the actual PR diff, the ONLY change is to `.github/workflows/extended.yml`. There are no Cargo.toml changes in this PR.

#### **The Command Changes**

**Before**:
```bash
cargo test --features backtrace,parquet_encryption --profile release-nonlto --test sqllogictests -- --include-sqlite
```

**After**:
```bash
# Build step
TEST_BIN=$(cargo build --profile release-nonlto --features backtrace,parquet_encryption --package datafusion-sqllogictest --test sqllogictests --message-format=json | ...)

# Run step (from datafusion/sqllogictest directory)
"$TEST_BIN" --include-sqlite
```

**Verification**: Both commands run **exactly the same tests**:
- Same profile: `release-nonlto`
- Same features: `backtrace,parquet_encryption`
- Same test binary: `sqllogictests` (the only test in the package)
- Same arguments: `--include-sqlite` flag passed to test binary
- The `--include-sqlite` flag includes both `test_files/` and `../../datafusion-testing/data/` test files

No tests are omitted. The only difference is `cargo test` vs. direct binary execution.

#### **Why This Might Be Faster** (Hypotheses, Ranked by Likelihood)

1. **CI runner variability** (most likely): GitHub-hosted runners have inconsistent performance. The 2+ hour and ~1 hour measurements may have been on different runners with different hardware. A controlled A/B test is needed.

2. **Package selection scope**: The old command has no `--package` flag. This workspace is **virtual** with no `default-members`, so Cargo selects **all 47 workspace members** (equivalent to `--workspace`). The new command uses `--package datafusion-sqllogictest`, scoping to just one package.
   - Impact: Cargo must resolve features (`backtrace,parquet_encryption`) and scan for matching `--test` targets across all 47 packages vs. 1 package.
   - However, actual *compilation* should be similar since `--test sqllogictests` filters the build to only matching targets.
   - **Likely saves minutes, not an hour.**

3. **Direct binary execution**: Bypasses `cargo test` runtime overhead (process spawning, test discovery, harness setup).
   - **Likely saves seconds.**

4. **Different code/caching across measurements**: If "before" and "after" timings were not from the same commit or CI cache state, the comparison is invalid.

**None of these adequately explain a full hour of savings.** The reviewer is right to be skeptical.

### Verification: No Tests Omitted

I verified that both workflows run identical tests by checking:

1. **Test binary**: Both build/run the `sqllogictests` test (confirmed it's the only test in `datafusion-sqllogictest` package via Cargo.toml)

2. **Features**: Both use `--features backtrace,parquet_encryption`

3. **Profile**: Both use `--profile release-nonlto`

4. **Arguments**: Both pass `--include-sqlite` to the test binary
   - This flag includes tests from `test_files/` directory
   - Plus additional tests from `../../datafusion-testing/data/` directory
   - Files starting with `sqlite` prefix are only included when this flag is set

5. **Test selection logic**: The `--include-sqlite` argument is parsed by the test binary's CLI (via clap), not by cargo, so both approaches behave identically

**Conclusion**: The new workflow runs exactly the same tests as the old workflow. No tests are omitted.

### The Honest Answer

**I don't fully understand why this would save so much time, and neither do you (or @nuno-faria).** This deserves investigation rather than speculation.

---

### Proposed Response to Reviewer

**Honest acknowledgment**:

> You're absolutely right to be hesitant. I'm confused too about why this saves so much time.
>
> The only change in this PR is the workflow command:
> - **Before**: `cargo test --profile release-nonlto --test sqllogictests -- --include-sqlite`
> - **After**: `cargo build --package datafusion-sqllogictest --test sqllogictests`, then run the binary directly from `datafusion/sqllogictest`
>
> **Potential reasons** (but none fully explain a 1-hour savings):  
> 1. `--package` flag helps Cargo skip unnecessary workspace-wide dependency checks
> 2. Direct binary execution bypasses `cargo test` harness overhead
> 3. Running from the correct directory (`datafusion/sqllogictest`) avoids path resolution issues
>
> **I agree with your suggestion**: Let's split the build and run into separate steps so we can see the timings. But more importantly, I want to be transparent: **I don't have a clear explanation for why this change would save an hour.** The most likely scenario is CI runner variability between measurements.
>
> The `--package` scoping in the new command does narrow Cargo's work (from scanning all 47 workspace members to just 1), but this should save minutes at most, not an hour.
>
> **Proposed next step**: Run a controlled A/B test — on the same commit, trigger both the old and new workflow commands to get comparable timing data. This will tell us whether there's a real improvement or just CI noise.

### Implementation Plan (Updated)

Following your suggestion to use explicit separate commands:

#### **Step 1: Build & Extract Test Binary Path**
```bash
# Build test binary and extract its path (single build, no redundancy).
TEST_BIN=$(cargo build --profile release-nonlto --features backtrace,parquet_encryption --package datafusion-sqllogictest --test sqllogictests --message-format=json | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p' | head -n 1)
if [ -z "$TEST_BIN" ]; then
  echo "Could not find sqllogictests test binary"
  exit 1
fi
```

**Rationale**: 
- Single `cargo build` (with `--message-format=json`) compiles the test binary and streams build metadata.
- Extracts executable path from JSON output; avoids brittle glob patterns.
- Explicit null check ensures early failure if binary not found.
- Isolates compilation time from test runtime for profiling.

#### **Step 2: Run Tests**
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

1. **Build timing**: How long does compilation take?
2. **Test timing**: How long does test execution take?

This gives us **visibility into performance**:
- Understand where time is spent (build vs. test)
- Confirm whether the workflow change actually improves performance
- If there is an improvement, understand which phase benefits
- Investigate why PR #19674 caused the original slowdown

---

### Next Steps (Investigation)

After this change is merged and workflow runs with split timing:

1. **Analyze timing data**:
   - Compare build time vs. test execution time
   - Look for patterns in CI logs

2. **Investigate the workflow difference**:
   - Why does `cargo test` take longer than `cargo build + direct execution`?
   - Is `cargo test` doing unnecessary dependency checks?
   - Is the `--package` flag making a difference?

3. **Compare with old approach**:
   - Optionally run both commands on same commit to measure difference
   - Document findings for future workflow optimization

4. **Address PR #19674 regression** (separate issue):
   - Once we have timing visibility, revisit whether struct validation changes need optimization
   - This is likely a separate investigation from the workflow command difference

---

## Revised Workflow Snippet

```yaml
- name: Build sqllogictest binary
  run: |
    # Single build with JSON output to extract test binary path.
    # Isolates compilation time for profiling.
    TEST_BIN=$(cargo build \
      --profile release-nonlto \
      --features backtrace,parquet_encryption \
      --package datafusion-sqllogictest \
      --test sqllogictests \
      --message-format=json | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p' | head -n 1)
    
    if [ -z "$TEST_BIN" ]; then
      echo "Error: Could not locate sqllogictests binary. Build may have failed."
      exit 1
    fi
    echo "TEST_BIN=$TEST_BIN" >> "$GITHUB_ENV"

- name: Run sqllogictest
  working-directory: datafusion/sqllogictest
  run: |
    # Run pre-built binary; isolate test runtime for profiling.
    # Test data resolved relative to datafusion/sqllogictest directory.
    "$TEST_BIN" --include-sqlite
```

---

## Summary

| Aspect | Status | Explanation |
|--------|--------|-------------|
| **Performance gain source** | ❓ Unknown | Only workflow changed; unclear why this saves ~1 hour |
| **Test coverage** | ✅ Verified | Both old and new run identical tests; nothing omitted |
| **Split build/run** | ✅ Agreed | Enables separate timing measurement |
| **Error checking** | ✅ Included | Fails fast if binary not found |
| **Timing clarity** | ✅ Enabled | CI logs will show build time ≠ test time |
| **Investigation needed** | ⚠️ Critical | Need data to understand the improvement |

**Key Insight**: We don't fully understand why changing from `cargo test` to direct binary execution saves significant time. The split build/test approach will provide the visibility needed to investigate.

**Recommendation**: Proceed with the split as suggested by @alamb to gather timing data, then investigate the root cause of the performance difference.

---

**Questions for continuation**:
- Should we run a side-by-side comparison (old command vs. new command) on the same commit to measure the actual time difference?
- Would it be helpful to add explicit timing echo statements in the workflow steps?
- Should we investigate whether `cargo test` was doing unnecessary rebuilds or workspace checks?
