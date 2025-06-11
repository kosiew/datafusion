# Simplified Filter Pushdown API

This document describes a simplified approach for pushing predicates through
physical plan nodes. The goal is to reduce the layering of abstractions and
make the pushdown behaviour easier to reason about.

## Core Data Structures

The new API introduces three main types located in
`datafusion/physical-plan/src/filter_pushdown_api.rs`:

- `PredicateWithSupport` – associates a predicate with either `Supported` or
  `Unsupported` status.
- `Predicates` – convenience wrapper around a collection of
  `PredicateWithSupport` values.
- `FilterPushdownResult<T>` – result of attempting to push predicates through a
  node; contains the pushed predicates, retained predicates and optionally an
  updated plan node of type `T`.

These types replace the previous combination of `PredicateSupport`,
`PredicateSupports`, `FilterDescription` and related helpers.

## Usage

Execution plan nodes can declare which predicates they support and return a
`FilterPushdownResult` describing what was pushed to children and what remains
local. The API is designed so that `with_*` methods create new objects while
`mark_*` methods transform existing ones. The focus is on keeping the flow
explicit and easy to follow.

This implementation is now integrated with the physical optimizer so that
`FilterPushdown` makes use of these structures when propagating predicates
through execution plans.

