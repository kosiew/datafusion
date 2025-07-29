## Memory Accounting API Evolution: Integrated Implementation Plan

### Executive Summary

This plan outlines a 6-phase, 12-week roadmap to integrate Arrow’s buffer‑level accounting into DataFusion, ensuring precision, performance, and compatibility.

---

### Current State Analysis

**DataFusion Today**

* Coarse‑grained tracking via `MemoryPool`/`MemoryReservation` at operator level
* Manual reservation for sorts, joins, aggregations; ignores intermediate RecordBatch buffers

**Arrow‑rs Memory API**

* Buffer‑level traits: `MemoryPool`/`MemoryReservation` on `Buffer`, `MutableBuffer`, `Bytes`
* RAII semantics: `reserve`, `resize`, `drop` for automatic accounting
* Feature‑gated to avoid overhead when disabled

---

### Strategic Phases & Timeline

| Phase                                                       | Weeks | Goals & Deliverables                                                            |
| ----------------------------------------------------------- | :---: | ------------------------------------------------------------------------------- |
| **1. Adapter Foundation**                                   |  1–2  | • Implement `DataFusionMemoryPool` adapter for `arrow_buffer::pool::MemoryPool` |
| • Map `reserve`/`resize` to DataFusion `MemoryReservation`  |       |                                                                                 |
| • Config flag `--enable-arrow-tracking`                     |       |                                                                                 |
| • Basic unit tests                                          |       |                                                                                 |
| **2. Integration Points**                                   |  3–4  | • Instrument RecordBatch creation in operators                                  |
| • Hook Parquet, CSV, JSON readers to use adapter pool       |       |                                                                                 |
| • Benchmarks to measure tracking overhead                   |       |                                                                                 |
| **3. Hybrid Tracking**                                      |  5–6  | • Build `HybridMemoryTracker` coordinating operator & buffer levels             |
| • Attribute shared buffers via zero-copy handling           |       |                                                                                 |
| • Add visualization tooling for memory breakdown            |       |                                                                                 |
| **4. Attribution & Reporting**                              |  7–8  | • Track buffer ownership and operator attribution                               |
| • Define streaming operator memory rules                    |       |                                                                                 |
| • Expose per-operator memory breakdown in metrics/dashboard |       |                                                                                 |
| • Integration tests for shared buffers                      |       |                                                                                 |
| **5. Performance Optimization**                             |  9–10 | • Profile end-to-end memory overhead (<10%)                                     |
| • Implement lazy accounting & batch reservations            |       |                                                                                 |
| • Performance regression suite                              |       |                                                                                 |
| **6. Advanced Features**                                    | 11–12 | • Auto-reservation based on buffer usage                                        |
| • Memory prediction for spill decisions                     |       |                                                                                 |
| • Forecasting for query planning                            |       |                                                                                 |
| • Operator-specific optimization strategies                 |       |                                                                                 |

### Key Design Decisions & Examples & Examples

#### 1. Adapter Pattern

```rust
struct DataFusionMemoryPool {
    inner: Arc<dyn df::MemoryPool>,
    reservation: df::MemoryReservation,
}

impl arrow_buffer::pool::MemoryPool for DataFusionMemoryPool {
    fn reserve(&self, bytes: usize) -> Result<(), ArrowError> {
        self.reservation.try_grow(bytes).map_err(|e| ArrowError::OutOfMemory(e.to_string()))
    }
    fn resize(&self, old: usize, new: usize) -> Result<(), ArrowError> {
        if new > old {
            self.reserve(new - old)
        } else {
            self.reservation.shrink(old - new);
            Ok(())
        }
    }
}
```

#### 2. Array Claiming

* Introduce `Array::claim(pool: &dyn MemoryPool)` default method to traverse and register all underlying buffers.
* Usage in DataFusion operators:

  ```rust
  let pool = context.arrow_memory_pool();
  batch.claim(pool);
  ```

---

### Success Metrics

1. **Accuracy**: Reported vs. actual memory <5% variance in microbenchmarks
2. **Performance**: Tracking overhead <10% in representative queries
3. **Compatibility**: Zero breaking changes; all existing tests pass
4. **Observability**: Live per‑operator breakdown via Prometheus metrics

---

### Testing & Observability

* **Unit & Integration Tests** for adapter, hybrid tracker, attribution scenarios
* **Benchmark Suite** measuring overhead across query patterns
* **Prometheus + Grafana** dashboards for real‑time memory usage
* **Alerting** on unexpected memory spikes or tracking failures

---

### Risk Mitigation

* **Performance Regression**: Feature flags (`--enable-arrow-tracking` / granularity levels) for quick rollback
* **Memory Overhead**: Lazy accounting, sampling modes, batch reservation
* **Compatibility**: Maintain default DataFusion `MemoryPool` API, exhaustive backwards‑compat tests
* **Complexity**: Phased rollout; each phase gated behind feature flags

---

### Long‑Term Considerations

* Propose higher‑level Arrow hooks (e.g., `RecordBatch::claim_all`) to simplify integration
* Explore reconciliation jobs to detect leaks or mismatches between Arrow & DataFusion pools
* Feedback loop: usage telemetry to refine sampling and prediction algorithms
