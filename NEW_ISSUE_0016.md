source: pr-21401_a
# Centralize benchmark download logic behind a shared helper in `bench.sh`

**Type:** Refactor / Developer Experience  
**Component:** `benchmarks/bench.sh`  
**Effort:** Medium  
**Origin:** Pre-existing technical debt, surfaced during review of PR #21401

---

## Summary

`benchmarks/bench.sh` has grown several independent, ad-hoc download call sites, each with its own tool choice (`wget` vs `curl`), retry strategy, progress output, and atomicity behavior. This inconsistency causes portability bugs (e.g., `wget` failing on stock macOS) and makes every new benchmark data source a copy-paste lottery. A single shared helper function would standardize this once.

---

## Current state: four inconsistent download patterns

| Call site | Tool | Retries | Progress | Atomic? | Notes |
|---|---|---|---|---|---|
| `data_tpch` — answers loop (line 618) | `wget` | `--tries=3` | `-q` (silent) | ❌ partial writes survive | `wget` absent on macOS by default |
| `data_tpcds` — zip download (line 661) | `wget` | `--tries=3` | None | ✅ single file | `wget` absent on macOS by default |
| `data_clickbench_1` — hits.parquet (line 761) | `wget` | None | Progress bar | ✅ `--continue` resumes | No retries |
| `data_clickbench_partitioned` — parallel (line 783) | `wget` in `xargs` subshell | None | `.` per file | ❌ partial set survives | No retries; hard to override tool |
| `data_imdb` — tgz (lines 896, 920) | `curl` | None | None | ❌ size-checks existing file only | Size mismatch check is post-hoc |

Key problems:
1. **`wget` is not installed on macOS by default.** `curl` is. Any call site that uses `wget` without a fallback breaks on a stock Mac developer machine.
2. **Partial downloads are not cleaned up atomically.** If a multi-file loop or a single large file is interrupted, the next run may skip the download entirely because a sentinel file already exists.
3. **Retry behavior is inconsistent.** Some paths retry 3 times; others have no retry at all; the parallel path has no retry per worker.
4. **Error messages and progress output differ** across call sites, giving a confusing UX.

---

## Proposed solution

Add a single `download_file` helper near the top of `bench.sh` (after the variable declarations):

```bash
# download_file <url> <destination>
# Downloads <url> to <destination> atomically using curl or wget (whichever is
# available), with 3 retries and a 30-second connect timeout.
# Writes to a temp file first and moves into place only on success.
download_file() {
    local url="$1"
    local dest="$2"
    local tmp="${dest}.tmp.$$"

    if command -v curl &>/dev/null; then
        curl --fail --retry 3 --retry-delay 2 --connect-timeout 30 \
             --location --progress-bar \
             -o "${tmp}" "${url}"
    elif command -v wget &>/dev/null; then
        wget --tries=3 --timeout=30 --waitretry=2 \
             -O "${tmp}" "${url}"
    else
        echo "ERROR: neither curl nor wget is available." >&2
        return 1
    fi

    mv "${tmp}" "${dest}"
}
```

Then replace each existing call site:

### `data_tpch` — answers loop
```bash
# Before
wget -q --timeout=30 --tries=3 -O "${TPCH_DIR}/answers/${OUT_FILE}" "${BASE_GH_URL}/${OUT_FILE}"

# After
download_file "${BASE_GH_URL}/${OUT_FILE}" "${TPCH_DIR}/answers/${OUT_FILE}"
```

### `data_tpcds` — zip download
```bash
# Before
wget --timeout=30 --tries=3 -O "${DATA_DIR}/datafusion-benchmarks.zip" https://...

# After
download_file "https://..." "${DATA_DIR}/datafusion-benchmarks.zip"
```

### `data_clickbench_1` — single large file
```bash
# Before
wget --continue ${URL}

# After
download_file "${URL}" "hits.parquet"
```
(Note: `--continue` / resume support can be layered in as an optional flag to `download_file` if desired.)

### `data_imdb` — tgz
```bash
# Before
curl -o "${imdb_temp_gz}" "${imdb_url}"

# After
download_file "${imdb_url}" "${imdb_temp_gz}"
```

### `data_clickbench_partitioned` — parallel workers

The parallel `xargs` call is the most complex. In a first pass, replace the inner `wget` with a call to `download_file` via a small wrapper exported to subshells:

```bash
export -f download_file
seq 0 99 | xargs -P${MAX_CONCURRENT_DOWNLOADS} -I{} bash -c \
    'download_file "https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_{}.parquet" \
                   "hits_{}.parquet" && echo -n "."'
```

---

## Atomicity fix for the TPC-H answers loop

The existence check for TPC-H answers currently only looks for `q1.out`:

```bash
FILE="${TPCH_DIR}/answers/q1.out"
if test -f "${FILE}"; then   # ← only q1.out is checked
```

This should be replaced with a full file-set validation before using `download_file`:

```bash
all_answers_present() {
    local dir="$1"
    for i in $(seq 1 22); do
        [ -f "${dir}/q${i}.out" ] || return 1
    done
    return 0
}

if all_answers_present "${TPCH_DIR}/answers"; then
    echo " Expected answers exist."
else
    echo " Downloading answers to ${TPCH_DIR}/answers"
    mkdir -p "${TPCH_DIR}/answers"
    for i in $(seq 1 22); do
        OUT_FILE="q${i}.out"
        download_file "${BASE_GH_URL}/${OUT_FILE}" "${TPCH_DIR}/answers/${OUT_FILE}"
    done
fi
```

This issue tracks that full-set validation separately from the helper refactor but both are best done in the same PR.

---

## Acceptance criteria

- [ ] A `download_file <url> <dest>` helper exists in `bench.sh`.
- [ ] Helper prefers `curl`, falls back to `wget`, and errors clearly if neither is present.
- [ ] Helper writes to a temp file and moves atomically on success.
- [ ] Helper retries at least 3 times with a timeout.
- [ ] All five existing call sites are migrated to use the helper.
- [ ] `data_tpch` validates all 22 answer files before skipping the download.
- [ ] `./bench.sh data tpch 1` works on a stock macOS machine without installing `wget`.
- [ ] `./bench.sh data tpcds` works on a stock macOS machine without installing `wget`.

---

## Related

- PR #21401 (introduced the `wget`-only TPC-H answers download that triggered this analysis)
- `benchmarks/queries/clickbench/update_queries.sh` (already uses a `curl`/`wget` fallback — good reference pattern)
