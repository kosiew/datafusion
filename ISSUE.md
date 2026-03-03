# Issue Summary: Severe Performance Pathology in `PushDownFilter` OptimizerRule

## Description

The `PushDownFilter` optimization rule in DataFusion exhibits pathologically slow performance during the planning phase of the `sql_planner_extended` benchmark. Profiling shows that a substantial portion of total planning CPU time is consumed inside this rule, with the majority of the time spent repeatedly computing expression types.

This results in disproportionate planning overhead and significantly degrades benchmark performance.

---

## Key Findings

### 1. Extreme Runtime Discrepancy

Profiling via `cargo samply` reveals:

* `push_down_filter` took **174 seconds**
* By comparison:

  * `single_distinct_aggregation_to_group_by` took **~164ms**
  * Other optimizer rules complete in milliseconds to low seconds

This indicates an orders-of-magnitude imbalance in rule execution time.

---

### 2. ~50% of Time Spent in Type Computation

Flamegraph and call tree analysis show that approximately **42–43% of total samples** are attributed to:

```
datafusion_expr::expr::Expr
  as datafusion_expr::expr_schema::ExprSchemable
  ::get_type
```

This indicates that nearly half of the entire planning time is spent computing expression types.

The profiling screenshots confirm:

* Wide, dense stacks dominated by:

  * `Expr::get_type`
  * `ExprSchemable::get_type`
  * `nullable`
  * `to_field`
  * `PredicateBoundsEvaluator::evaluate_bounds`
* Deep and repeated recursion through expression schema logic

Type resolution appears to be recomputed extensively during filter pushdown traversal.

---

### 3. Deep Recursion Through Expression Schema Logic

The stack chart shows repeated patterns of:

```
Expr::get_type
  -> ExprSchemable::get_type
     -> nullable
        -> to_field
           -> PredicateBoundsEvaluator::evaluate_bounds
```

These calls appear multiple times within nested optimizer rewrites, suggesting:

* Lack of memoization or caching of expression type information
* Repeated schema derivation across identical subtrees
* Potential quadratic or worse behavior depending on logical plan shape

---

### 4. Optimizer Rewrite Amplification

`PushDownFilter` is implemented as an `OptimizerRule::rewrite`, and profiling indicates heavy time spent inside rewrite recursion, including:

* `TreeNode::transform_up`
* `TreeNode::map_children`
* Expression cloning
* Logical plan rewriting

The combination of:

* Deep logical plan traversal
* Repeated type computation
* Expression cloning
* Predicate bounds evaluation

appears to amplify computational cost dramatically.

---

## Reproduction Steps

```
RUST_LOG=info cargo samply --profile=release-nonlto --bench sql_planner_extended -- --nocapture --sample-size 10
```

Profiling data available at:
[https://github.com/Omega359/arrow-datafusion/tree/profile_optimize](https://github.com/Omega359/arrow-datafusion/tree/profile_optimize)

---

## Impact

* `PushDownFilter` dominates planner CPU usage
* Planning time scales poorly
* Benchmark performance is heavily skewed by type resolution overhead
* Planner performance does not scale gracefully with logical plan complexity

This represents a critical planning-phase bottleneck.

---

## Expected Behavior

* Optimizer rules should complete in milliseconds to low seconds
* Type computation should not dominate planning
* Logical plan rewrites should not exhibit near-exponential behavior
* Repeated expression schema evaluation should be avoided

---

## Suspected Root Causes

Based on profiling evidence:

1. Repeated invocation of `ExprSchemable::get_type` without caching
2. Deep recursive rewrite passes in `PushDownFilter`
3. Predicate bounds evaluation triggering additional type derivations
4. Lack of memoization for schema/type results on expression subtrees

---

## Conclusion

`PushDownFilter` exhibits a severe performance pathology in which approximately half of total planning time is spent recomputing expression types. The rewrite logic combined with repeated schema derivation leads to disproportionate CPU usage in the `sql_planner_extended` benchmark.

Addressing redundant type computation and reducing recursive rewrite overhead will likely yield significant planner performance improvements.
