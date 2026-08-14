source: memcalc-02-23393a
# Validate memory accounting against allocator measurements

## Problem

`size()` / `mem_size()` estimates are being corrected, but their relationship to allocator-reported memory is unverified. Current measurements can conflate DataFusion-owned retained allocations with allocator rounding, caches, and heap fragmentation.

## Why it matters

Without an allocator comparison, an apparently accurate accounting formula may still materially diverge from live allocation behavior. Conversely, allocator fragmentation can make a correct retained-capacity estimate look wrong.

## Invariant / desired behavior

For a controlled workload with known ownership and minimal fragmentation, DataFusion's accounting reports the owned retained allocation total. Allocator measurements are a validation signal, not a replacement contract: differences caused by allocator metadata, size-class rounding, caches, or fragmentation are identified separately.

## Proposed direction

After the current `mem_size` accuracy work stabilizes, build a small reproducible measurement harness. Run identical workloads with the default allocator and mimalloc, capture DataFusion-reported bytes plus allocator live/allocated statistics, and document the comparison. Prefer fresh-process, deterministic workloads that allocate, retain, and release known structures; use an environment that minimizes or eliminates fragmentation if available.

## Scope

### In
- Define which allocator statistic corresponds most closely to DataFusion retained-allocation accounting.
- Add or document a reproducible comparison harness for representative accounting paths.
- Compare default allocator and mimalloc results.
- Separate expected allocator overhead/fragmentation from accounting defects.

### Out
- Making `size()` equal process RSS, reserved virtual memory, or allocator heap footprint.
- Changing production accounting formulas solely to match allocator-specific behavior.
- Broad allocator-performance benchmarking.

## Acceptance criteria

- [ ] A documented workload has deterministic allocation/retention phases and reports DataFusion accounting plus allocator statistics.
- [ ] The workload runs with the default allocator and mimalloc.
- [ ] Results state the selected allocator metric and explain expected differences from DataFusion accounting.
- [ ] At least one low-fragmentation/fresh-process run establishes a baseline for comparing accounting accuracy.
- [ ] Any material unexplained divergence produces a focused follow-up issue with an ownership boundary and reproducer.

## Tests / verification

- Manual/reproducible benchmark or integration harness in isolated fresh processes.
- Repeat runs for each allocator; record allocator version, platform, workload parameters, and reported metrics.
- Verify allocation, retention, and release phases independently before interpreting totals.

## Notes / open questions

- Which allocator metrics are portable enough for CI versus manual benchmarking?
- Can a supported allocator/test configuration provide an effectively non-fragmenting baseline, or should a purpose-built tracking allocator be the reference?
