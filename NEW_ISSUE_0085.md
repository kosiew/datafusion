source: pr-22839_a
# Make one-shot CLI input sources explicit in object-store registration

## Summary

DataFusion CLI support for `/dev/stdin` currently models stdin as a `stdin://` object store. This works well for routing reads through the existing listing/object-store path, but the ownership/lifecycle invariant is implicit:

> stdin is a one-shot process stream. Once consumed, later registrations must not silently replace or mutate the backing data for already-created tables.

This invariant should be encoded directly in CLI object-store registration rather than relying on helper behavior or documentation.

## Motivation

A pipe can only be consumed once. If each `stdin://` object-store lookup reads from `std::io::stdin()` independently, a second registration in the same CLI session will read EOF and may replace the existing store with an empty object. Existing tables that resolve through the same registered URL can then observe changed data.

This makes `/dev/stdin` different from ordinary object stores:

- local/S3/GCS/HTTP stores are stable references to external data
- stdin is process-local, destructive, and session-scoped
- re-registering it must be guarded or made idempotent

Making this explicit improves correctness now and gives a clear abstraction for future pipe-like sources.

## Current behavior / context

Observed in PR apache/datafusion#22839 while reviewing cumulative commits `6eb3866687^..9f68e244c3`.

Touched code:

- `datafusion-cli/src/exec.rs`
  - rewrites `/dev/stdin` to `stdin:///stdin.<ext>` before registering object store
- `datafusion-cli/src/object_storage.rs`
  - dispatches `stdin` scheme to stdin helper
- `datafusion-cli/src/object_storage/stdin.rs`
  - `StdinUtils::object_store(url)` reads all of process stdin into a new in-memory store

The helper is clean locally, but lifecycle semantics belong at the registration/session layer because replacing a registered store affects all plans/tables that reference the same URL.

## Problem example

Given piped CSV input:

```bash
printf 'a,b\n1,foo\n2,bar\n' | datafusion-cli -q -c "
CREATE EXTERNAL TABLE t STORED AS CSV LOCATION '/dev/stdin'
  OPTIONS ('format.has_header' 'true');
SELECT count(*) FROM t;
CREATE EXTERNAL TABLE t2 STORED AS CSV LOCATION '/dev/stdin'
  OPTIONS ('format.has_header' 'true');
SELECT count(*) FROM t;
"
```

Bad outcome seen during review:

```text
2
0
```

The second `CREATE EXTERNAL TABLE` consumes EOF and replaces the registered `stdin://` object store, so the original table no longer sees its original data.

## Desired invariant

For one CLI session:

1. The first `/dev/stdin` registration may consume stdin and buffer it.
2. Later `/dev/stdin` registrations must not silently replace the previous buffered data.
3. Existing tables backed by stdin must remain stable after creation.
4. If reuse is unsupported, the CLI should fail clearly and early with an actionable error.

## Possible approaches

### Option A: reject subsequent stdin registrations

Track whether stdin has already been consumed in the CLI session. On a second `/dev/stdin` registration, return an error such as:

```text
Standard input has already been consumed in this session; /dev/stdin can only be registered once
```

Pros:

- simple
- matches docs saying stdin can only be consumed once
- avoids ambiguity when table formats/options differ

Cons:

- users cannot create two tables over the same buffered stdin, even if they intend to reuse it

### Option B: cache and reuse the first buffered store

Store the first buffered stdin object store in session/runtime state and return the same store for later `stdin://` lookups.

Pros:

- stable, idempotent registration
- allows repeated references to the same stdin data

Cons:

- needs clear behavior for different rewritten paths (`stdin.csv`, `stdin.json`, `stdin.parquet`)
- needs policy for conflicting formats/options
- may be more surprising if a user expects a new read

### Option C: allocate unique session-local stdin object URLs and guard source consumption

Separate the destructive source (`process stdin`) from buffered objects. First use consumes stdin and creates a named buffered object. Later uses either refer to the same object explicitly or fail.

Pros:

- strongest model for future one-shot sources
- clarifies source vs object lifecycle

Cons:

- more design work
- likely larger than needed for this PR

## Suggested implementation direction

Start with Option A unless there is a strong user need for repeated table creation from the same stdin buffer.

A small design could be:

- add a session-scoped stdin registration guard/cache in CLI-specific context or object-store registration path
- keep `StdinUtils` focused on path rewrite and buffering mechanics
- have `register_object_store_and_config_extensions` or a CLI wrapper detect the `stdin` scheme and enforce the guard before calling `get_object_store`
- never call `ctx.register_object_store(url, store)` for stdin after it has already been registered unless intentionally reusing the same store

## Tests to add

Add CLI integration or targeted unit coverage for:

1. first `/dev/stdin` table reads CSV successfully
2. second `/dev/stdin` table in same session does not mutate the first table
3. chosen policy is explicit:
   - if rejecting: assert clear error on second registration
   - if reusing: assert both first and second queries see original row count
4. existing CSV/JSON/Parquet stdin tests continue to pass

Example regression shape:

```sql
CREATE EXTERNAL TABLE t STORED AS CSV LOCATION '/dev/stdin'
  OPTIONS ('format.has_header' 'true');
SELECT count(*) FROM t;
CREATE EXTERNAL TABLE t2 STORED AS CSV LOCATION '/dev/stdin'
  OPTIONS ('format.has_header' 'true');
SELECT count(*) FROM t;
```

Expected result should not be `2` then `0`.

## Acceptance criteria

- `/dev/stdin` consumption is guarded or cached at a session-aware layer.
- Repeated `/dev/stdin` registration cannot silently replace previously buffered data.
- Error message, if rejecting repeat usage, explains stdin is one-shot.
- Tests cover repeated registration behavior.
- Documentation matches the implemented policy.

## Scope notes

This is a CLI-specific lifecycle/refactor issue. It should not require changing DataFusion core datasource semantics unless a reusable one-shot object-store abstraction is intentionally introduced.
