# Issue Description

# Issue Summary: Integration of `CastColumnExpr` into `PhysicalExprAdapter`
## Context
The proposal revolves around integrating `CastColumnExpr` into the existing `PhysicalExprAdapter` within the DataFusion project.
Discussions have highlighted issues and potential improvements in handling type casting in query expressions.
## Implementation Steps Proposed by @kosiew
1. **Introduce `CastColumnExpr`:**
Add the new expression type with evaluation semantics for arrays and scalars.
Address nullable/return-field fixes and create unit tests in the physical expression crate.
2. **Extend Utilities for `CastColumnExpr`:**
Update equivalence tracking, interval reasoning, and cast-unwrapping simplifications so that the optimizer recognizes the new node.
3. **Serialization Support:**
Implement serialization/deserialization for `CastColumnExpr` via protobuf schema.
Include round-trip tests for distributed execution.
4. **Update Pruning Logic:**
Extend `rewrite_expr_to_prunable` to handle `CastColumnExpr` in statistics-based pruning.
5. **Refactor Schema Rewriter:**
Conduct structural cleanup and introduce new helper routines for clarity and efficiency.
6. **Finalize Adapter Changes:**
Switch the adapter to produce `CastColumnExpr`.
Add behavior tests focusing on error handling and nullable cases.
## Ongoing Discussions and Opinions
**Integration with `CaseExpr`:**
@alamb suggests that integrating the logic for struct casting into `CaseExpr` could offer a more elegant, long-term solution.
The discussion leans toward merging `CastColumnExpr` with `CaseExpr` to streamline functionality.
**Current Status:**
@alamb is working on the integration but is currently stalled.
@adriangb has expressed interest in reviving the work and has linked to relevant pull requests showcasing issues related to the need for integration.
## Challenges Identified
The existence of special cases (like `CaseColumnExpr`) is complicating implementation and causing issues in other areas.
There is general agreement among contributors that consolidation of casting logic could improve code maintainability and functionality.
## Next Steps
Review the proposed plan for integrating `CastColumnExpr` and evaluate its feasibility.
Consider resuming discussions on merging with `CaseExpr` or exploring alternative implementations to enhance robustness in expression handling.
