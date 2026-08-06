# SemVer Review Guide

Use this guide when a PR changes a publishable workspace crate. The review target
is compatibility with the PR base branch, not only whether the implementation
works on its own. It supplements the repository's
[API health policy](docs/source/contributor-guide/api-health.md).

## Core rule

A release compatible under SemVer must not make a supported downstream program
stop compiling or change an established contract unexpectedly. `cargo-semver-checks`
is useful evidence for Rust API compatibility; it does not cover SQL behavior,
serialized data, configuration, ABI, or all generated-code contracts.

## First checks

1. Identify changed publishable crates:

   ```bash
   ci/scripts/changed_crates.sh changed-crates apache/main
   ```

2. Inspect the cumulative diff and the crate versions relative to the base
   branch. A version unchanged from the base cannot carry a breaking change.
3. Run the SemVer check for each changed published crate:

   ```bash
   ci/scripts/changed_crates.sh semver-check apache/main <crate>...
   ```

4. State the old contract, new contract, affected users, and migration path.
   Do not treat a passing check as proof that no compatibility contract changed.

## What is normally breaking

| Change                                                                                       | Why it breaks                                                                             |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Remove, rename, narrow visibility of public item                                             | Existing imports/calls fail to compile.                                                   |
| Change public function, method, type alias, generic bound, or return type                    | Existing callers no longer type-check or observe different API behavior.                  |
| Add required public trait method; alter object safety                                        | Downstream trait implementations or trait objects fail to compile.                        |
| Remove public enum variant or add a variant to an exhaustive public enum                     | Existing `match` expressions may fail to compile.                                         |
| Change public struct fields, layout-sensitive FFI types, or generated public types           | Construction, matching, ABI, or serialization users may break.                            |
| Add a protobuf `oneof` arm generating a public Rust enum variant                             | Downstream exhaustive matches on the generated enum fail.                                 |
| Change SQL result values, types, nullability, ordering, errors, accepted syntax, or defaults | SQL clients depend on observable behavior, even if Rust APIs compile.                     |
| Change persisted/protocol encoding, schema, metadata, or FFI/ABI behavior incompatibly       | Old readers/writers, saved plans, plugins, or external clients fail or misinterpret data. |
| Remove/rename config, change default/units/meaning, or reject formerly valid values          | Deployments change behavior at upgrade.                                                   |

## Usually compatible

Accept when evidence confirms the change is additive and preserves old behavior:

- Private/internal refactor; no public or observable contract changes.
- New public function, type, or optional configuration with safe default.
- New trait method with a behavior-preserving default implementation.
- New enum variant only when the enum is not publicly exhaustively matchable, or
  the API explicitly provides a forward-compatible non-exhaustive boundary.
- Bug fix that restores documented behavior and does not require downstream
  source changes. Document material behavior corrections.
- Additive protobuf field/`oneof` data only when every exposed consumer boundary
  is forward-compatible. Generated public Rust enums require separate review.

"Additive" is not enough by itself: prove current callers retain compilation
and behavior.

## Decision table

### Reject or request redesign

Reject a compatibility break when any applies:

- The release version does not permit it.
- A compatible design exists: additive API, default trait method, deprecated
  shim, extension point, or backward-compatible encoding.
- The PR changes behavior incidentally, without an explicit user contract.
- Migration requires every downstream user to change but the new capability is
  not worth that cost.
- The break is hidden in generated code, a re-export, a default, a protocol, or
  a feature flag rather than called out directly.
- Tests cover only new behavior, not old-client/old-data compatibility.

Example: reject a new public generated protobuf `oneof` variant in a
non-major-compatible release. Its wire encoding may be additive, but the
public generated Rust enum becomes non-exhaustive for downstream matches.

### Accept with changes

Accept only after the PR provides all applicable items:

- Release/version strategy that permits the break.
- Exact old and new API/behavior, including affected crate(s).
- Why a compatible alternative is insufficient.
- Concrete migration/replacement path; deprecation where practical.
- `api-change` label for breaking Rust API changes and an Upgrade Guide entry
  for non-trivial changes.
- For breaking SQL changes, PR description states old/new behavior, ideally
  with example query results.
- Tests at each changed boundary: compile/API, SQL/runtime, serde/proto,
  FFI/ABI, config, or schema.
- Explicit compatibility direction: old producer -> new reader, new producer
  -> old reader, old client -> new server, and/or new client -> old server.

A major-version boundary permits a break; it does not justify an accidental,
undocumented, or needlessly broad one.

### Accept

Accept a detected SemVer change only when it is intentional, justified, scoped,
and accompanied by the evidence above. Accept no detected break merely because
CI reports it: the check identifies a contract change; review decides whether
that contract may change in this release.

Prefer deprecation over removal when it provides a viable transition. Mark the
replacement and deprecation version. DataFusion keeps deprecated methods for at
least six major versions or six months, whichever is longer.

## Boundary-specific review

### Rust public API

Check public modules, re-exports, type aliases, macros, constructors, trait
objects, trait implementors, enum exhaustiveness, public fields, feature-gated
APIs, and generated sources. Compile a small downstream-style use case when
uncertain.

For traits, distinguish callers from implementors. Adding a default method may
be source-compatible for implementors, but changing semantics or object safety
can still be breaking.

### Protobuf and generated models

Review both contracts:

1. Wire compatibility: unknown fields/variants, required fields, tag reuse,
   old/new decoding, JSON mapping, and persisted plans.
2. Generated Rust API: public structs/enums, exhaustive matches, constructors,
   re-exports, and serde behavior.

Never assume Protobuf's additive wire rules imply Rust SemVer compatibility.
Use new tags; never reuse a removed tag for a different meaning.

### SQL, plans, schemas, and execution

Check values, errors, data types, nullability, field/schema metadata, ordering,
partitioning, optimization eligibility, and plan serialization. A correction to
previously undocumented behavior can still be a material upgrade change; state
why it is a bug fix and test the prior edge.

### Configuration and operations

Check names, aliases, defaults, units, validation, precedence, environment
variables, and rollout/rollback behavior. A default change is breaking when a
user who does nothing after upgrade gets materially different behavior.

### FFI, plugins, and persisted formats

Treat layouts, vtables, symbols, ownership rules, callbacks, capability
negotiation, and serialized schemas as public contracts. Prefer versioned or
negotiated extensions over changing an existing representation in place.

## Evidence checklist

- [ ] Changed publishable crates identified.
- [ ] Base and PR versions compared.
- [ ] `cargo-semver-checks` result recorded or its limitation explained.
- [ ] Public/generated/re-exported API reviewed.
- [ ] Observable behavior and compatibility directions reviewed.
- [ ] Compatible alternative considered.
- [ ] Migration and release-note/upgrade-doc needs identified.
- [ ] Regression tests prove the relevant old/new boundary.

## Review comment template

```md
Severity: <high/medium>

This changes <old contract> to <new contract> in <crate/boundary>. Existing
<downstream users/clients/data> can <compile fail / fail to decode / observe
changed behavior>. The current release version <does/does not> permit that.

<Reject: use compatible direction.>
<Accept conditionally: state version strategy, migration, docs, and tests.>
```
