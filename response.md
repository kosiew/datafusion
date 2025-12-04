# Review: Recursive Substrait PR — e7308017f^..7a4f6c02d

This document answers the challenge questions and enumerates alternatives for the changes introduced in the commits mentioned above. It focuses on interoperability, schema stability, producer/consumer semantics, and test coverage.

---

## 1 • Design of the Recursive Type URLs 🔗

### Challenge Questions — short answers
- Why custom type URLs? — To express DataFusion-specific recursive constructs using Substrait's extension mechanism while keeping the core Substrait plan stable.
- Are the type URLs stable and how will compatibility be handled? — Not implicitly. Stability must be enforced by either versioned type URLs or an agreed extension registry.
- How do external consumers behave when seeing unknown type URLs? — They either ignore unknown extensions (which loses semantics) or error out (safer but less interoperable). Behavior is consumer-specific unless documented.

### Analysis & Trade-offs
- Using custom URLs is pragmatic for rapid development but risks portability and future collisions.
- Without versioning or documentation, evolution of these URLs (or their payloads) will create breaking changes for third-party consumers.

### Alternatives & Recommendations
- Use versioned type URLs, e.g., `com.datafusion.recursive.v1.RecursiveQuery` to make intent and evolution explicit.
- Coordinate with Substrait maintainers / the community and align with any official recursion spec where possible.
- Optionally wrap payloads in a small stable envelope (e.g., ExtensionMessage { version, type_url, payload }) to normalize consumer handling and ease future upgrades.

---

## 2 • Metadata Encoding / Decoding 🧩

### Challenge Questions — short answers
- Are prost messages the simplest long-term representation? — prost/protobuf is robust and efficient but requires careful versioning and tooling; text-based encodings are easier to inspect.
- Should versioning be included in metadata schemas? — Yes, explicitly.
- Strict validation vs defaults — Strict validation catches problems early, but permissive modes help with forward compatibility and interoperability.

### Analysis & Trade-offs
- Binary protobuf metadata is compact and typed, suitable for production, but has downsides for debugging and ad-hoc evolution.
- Lack of versioning increases the risk of consumer breakage as fields are added or semantics change.

### Alternatives & Recommendations
- Add a `schema_version` or `metadata_version` field to every recursion metadata message, following semantic-major awareness.
- Keep protobuf as canonical but provide optional JSON round-tripping (for logging or human inspection) in tooling/lint checks.
- Provide two runtime validation modes: strict (CI/tests) and permissive runtime parse with warnings/logging for unknown or optional fields.
- For experimentation, consider an extensible `map<string, string>` fallback to hold ad-hoc fields while keeping the stable typed message for primary fields.

---

## 3 • Producer-Side RecursiveQuery Handling (encoding/substrait producer) 🏭

### Challenge Questions — short answers
- Why exactly two inputs? — Likely intended to represent a base term and a recursive iteration term — a simple canonicalization of recursion. This is limited for more complex recursive patterns.
- Is `emit_kind = Direct` correct for all downstream engines? — Not necessarily. Different consumer engines may expect different append/emit semantics; `Direct` is one reasonable choice but document the semantics explicitly.
- Could projection wrapping or structural additions impact optimizers? — Yes. Added projections or structural changes can prevent downstream optimizers from recognizing ideal patterns or canonical forms.

### Analysis & Trade-offs
- The two-input model is straightforward but might not represent multi-base or mutually recursive scenarios.
- Emission semantics should be explicit; engines interpreting `Direct` differently will lead to inconsistent results.

### Alternatives & Recommendations
- Allow multiple static/base inputs or provide an explicit list-of-base-terms representation for broader expressivity.
- Make `emit_kind` more explicit and extensible. Document its concrete meaning and expected consumer behavior.
- Avoid unnecessary projection/permutation changes in the producer translation. If structural changes are required, annotate them clearly so consumers and optimizers can reason about intent.
- Consider producing a canonical high-level `RecursiveCTE` rel node where Substrait permits extension of rel node type rather than shoehorning recursion into a two-operand structure.

---

## 4 • Consumer-Side RecursiveQuery Reconstruction (substrait → datafusion logical plan) 🔄

### Challenge Questions — short answers
- How robust is decoding when encountering malformed or partially supported constructs? — Decoding should be robust: detect problems early and fail clearly, or fall back to permissive mode with warnings.
- Can unsupported children cause silent semantic drift? — Yes. If unknown fields are ignored, consumer semantics may change silently. This is a risk without strict validation or versioned schemas.
- Is cycle detection or recursion safety validated during plan reconstruction? — Ideally yes — consumers should check for malformed recursion that could cause infinite loops or invalid plans.

### Analysis & Trade-offs
- A permissive consumer that ignores unknown fields may construct a plan that looks plausible but is semantically wrong.
- Failing on unknown constructs is safer but prevents interoperability with producers that provide optional metadata.

### Alternatives & Recommendations
- Implement a strict decode path for CI/test time and a permissive decode path for internal tooling/runtime with clear diagnostics.
- Validate recursion safety: check cycles, ensure termination hints are present or generate analysis warnings when a plan could loop infinitely.
- Provide clear error semantics for unsupported children instead of silently dropping them.
- Offer a plug-in mechanism for custom extension handlers so a consumer can opt-in to understanding specific producer extensions.

---

## 5 • Recursive Work-Table Scan Encoding (ReadRel.advanced_extension) 🧾

### Challenge Questions — short answers
- Is embedding work-table metadata in ReadRel.advanced_extension the cleanest approach? — It's pragmatic and lightweight but couples internal producer concepts (work tables) to read/scan constructs.
- Are table-name collisions possible? — Yes, if names are not namespaced or globally unique, collisions across queries or CTEs are possible.
- Does this expose internal DataFusion details too directly? — It can; embedding implementation detail risks leaking internal semantics and making future refactors harder.

### Analysis & Trade-offs
- Embedding metadata into ReadRel.advanced_extension is simple, but it blurs the boundary between portable plan constructs and engine-specific state.
- Table name collisions are likely if a simple name is used — namespacing or UUIDs are safer.

### Alternatives & Recommendations
- Use a dedicated Substrait rel node type for work-table scans in recursive contexts (cleaner separation of concerns).
- If embedding in ReadRel.advanced_extension remains necessary, adopt explicit namespacing (e.g., `cte:<query_id>:worktable:<name>`) or embed a UUID to avoid collisions.
- Keep the information representation small and stable; read-side consumers should prefer a single canonical way (id + name + schema) to identify work-table references.

---

## 6 • Consumer Resolution Logic (catalog lookup overrides) 🔎

### Challenge Questions — short answers
- Is overriding catalog lookup safe when advanced_extension indicates recursion? — It's risky: without clear scoping, it can hide legitimately user-defined tables.
- Could it shadow user tables? — Yes, a poorly-scoped override could lead to unintended shadowing.
- How are nested/shadowed CTE scopes differentiated? — They must be articulated explicitly in the plan (e.g., via query-scoped IDs, nested scopes, or unique identifiers for CTEs) — otherwise ambiguity will arise.

### Analysis & Trade-offs
- Blindly overriding catalog lookup reduces safety — legitimate user tables may be mistaken for work-table scans.
- Consumers need well-defined scoping rules and a mechanism to disambiguate identical names across nested scopes.

### Alternatives & Recommendations
- Make CTE scope explicit in the Substrait plan — include a `scope_id` or `cte_id` for every work-table reference.
- Prefer UUIDs or globally unique identifiers over simple names to prevent accidental shadowing.
- If a consumer must override catalog lookup behavior, do so only when `is_recursive_work_table` metadata is present and a recommended precedence list is observed (e.g., check local CTE-first, then catalog-last).

---

## 7 • Test Coverage & Semantics Validation ✅

### Challenge Questions — short answers
- Are deep/mutual recursive CTEs tested? — If not yet, they should be added; mutual recursion is a common advanced case.
- Are semantic equivalence and structural equivalence validated? — Tests should ensure both round-trips (DataFusion→Substrait→DataFusion) preserve semantics; structural equivalence is often insufficient.
- Do tests check optimizer behaviour or performance for recursive plans? — Tests should also include optimizer/transform pass interactions and performance checks for large recursion depths.

### Recommended Test Matrix
- Unit tests for: basic single-base recursion, multiple-base terms, mutual recursion, large-depth recursion termination checks, malformed payload handling.
- Round-trip tests: DataFusion plan → Substrait → DataFusion ensures semantic equality by executing both plans on the same data and comparing results.
- Edge/mutation tests: ensure unknown/extra fields do not create silent drift and that version mismatches are flagged.
- Fuzz/differential tests (recommended): generate random recursive plans to ensure serialization and deserialization robustness.
- Performance/regression tests: generate large recursive plans and check for regressions in planning/execution time.

### Example test suggestions
- Add SQL Logic Tests (SLT) that use WITH RECURSIVE with multi-base and mutual recursion; assert output and explain plans.
- Add round-trip Substrait tests that create a Substrait plan with recursion metadata and verify DataFusion can restore it and produce identical results.

---

## Conclusion & Actionable next steps ✍️

1. Add `schema_version` fields to recursion metadata messages and make type URLs carry versioning.
2. Document all type URLs used in the repo and propose coordination with Substrait maintainers for a shared extension registry.
3. Increase test coverage: add mutual recursion tests, round-trip serialization tests, safety checks (cycle/termination), and fuzz tests for recovery from malformed metadata.
4. Consider a dedicated rel node or clearer wrapper message for recursion constructs to separate engine internals from portable semantics.

If you'd like, I can:
- open a follow-up PR to add schema_version and types to the Substrait extension protos,
- add a set of SLT and round-trip tests for recursion, or
- propose a public extension registry entry for the type URLs used.

---

Produced by: a careful review of e7308017f^..7a4f6c02d and the recursive-substrait design space.
