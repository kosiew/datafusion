# String Interning Investigation: Complete Documentation Index

This index provides a comprehensive guide to all documentation created during the historical investigation of DataFusion's string interning system.

---

## Quick Start (Read These First)

1. **[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** ⭐ **START HERE**
   - High-level overview of the problem and solution
   - For: Decision makers, quick understanding
   - Time to read: 5 minutes

2. **[STRING_INTERNING_QUICK_REFERENCE.md](STRING_INTERNING_QUICK_REFERENCE.md)**
   - Timeline table, key milestones
   - Comparison of design choices
   - For: Quick lookup, timeline at a glance
   - Time to read: 3 minutes

---

## Detailed Reference Materials

### Historical Investigation
3. **[INVESTIGATION_ANSWERS.md](INVESTIGATION_ANSWERS.md)**
   - Direct answers to all original questions
   - Question 1: When was `PhysicalExprAdapter` introduced?
   - Question 2: When was `FormatOptions`/`CastOptions` handling added?
   - Question 3: What changed in Arrow/DataFusion to introduce the lifetime mismatch?
   - Question 4: Was there a time when deserialization worked without string interning?
   - For: Understanding the complete history
   - Time to read: 10 minutes

4. **[TIMELINE_PROTO_STRING_INTERNING.md](TIMELINE_PROTO_STRING_INTERNING.md)**
   - Phase-by-phase evolution with commit details
   - Explains why each phase was necessary
   - Shows the causal chain of events
   - For: Deep historical understanding, decision-making
   - Time to read: 15 minutes

### Technical Deep Dive
5. **[STRING_INTERNING_IMPLEMENTATION.md](STRING_INTERNING_IMPLEMENTATION.md)**
   - Current implementation details with code examples
   - Cache structure, insertion logic, thread safety
   - Memory behavior analysis (best/typical/worst case)
   - Performance considerations and optimization opportunities
   - For: Developers working with the code
   - Time to read: 20 minutes

6. **[PROTO_STRING_INTERNING_ISSUE.md](PROTO_STRING_INTERNING_ISSUE.md)**
   - Original problem statement and context
   - Detailed explanation of the lifetime constraint
   - Comparison of alternative approaches
   - Current solution walkthrough
   - For: Understanding the problem from first principles
   - Time to read: 15 minutes

---

## Document Map by Use Case

### "I need to understand what happened"
→ Read in order:
1. EXECUTIVE_SUMMARY.md (5 min)
2. INVESTIGATION_ANSWERS.md (10 min)
3. TIMELINE_PROTO_STRING_INTERNING.md (15 min)

### "I need to implement changes related to this"
→ Read in order:
1. EXECUTIVE_SUMMARY.md (5 min)
2. STRING_INTERNING_IMPLEMENTATION.md (20 min)
3. STRING_INTERNING_QUICK_REFERENCE.md (3 min)

### "I need to debug/troubleshoot the cache"
→ Read:
1. STRING_INTERNING_IMPLEMENTATION.md (focus on "Error Handling" section)
2. PROTO_STRING_INTERNING_ISSUE.md (focus on "Current Solution" section)

### "I need to optimize or improve this"
→ Read:
1. STRING_INTERNING_IMPLEMENTATION.md (focus on "Performance Considerations" section)
2. INVESTIGATION_ANSWERS.md (focus on "Why Other Solutions Don't Work" section)

### "I need to explain this to someone else"
→ Share:
1. EXECUTIVE_SUMMARY.md (for overview)
2. STRING_INTERNING_QUICK_REFERENCE.md (for quick facts)
3. Specific detailed docs based on their questions

---

## Key Topics by Document

### Arrow 57.0.0 Upgrade (The Root Cause)
- **EXECUTIVE_SUMMARY.md** - "The Problem in One Picture"
- **INVESTIGATION_ANSWERS.md** - "Question 3: What changed in Arrow"
- **TIMELINE_PROTO_STRING_INTERNING.md** - "Phase 3: Arrow 57.0.0 Upgrade"

### PhysicalExprAdapter (The Enabler)
- **INVESTIGATION_ANSWERS.md** - "Question 1: When was PhysicalExprAdapter introduced"
- **TIMELINE_PROTO_STRING_INTERNING.md** - "Phase 1: Physical Expression Adapter Introduction"

### CastColumnExpr (The Expression Type)
- **INVESTIGATION_ANSWERS.md** - "Question 2: When was FormatOptions/CastOptions handling added"
- **TIMELINE_PROTO_STRING_INTERNING.md** - "Phase 2: CastColumnExpr Introduction"

### Format Options Serialization (The Exposure)
- **INVESTIGATION_ANSWERS.md** - "Question 2 Phase B" and "Question 4"
- **TIMELINE_PROTO_STRING_INTERNING.md** - "Phase 4: Cast Column Format Options in Protobuf"

### String Interning Cache (The Solution)
- **PROTO_STRING_INTERNING_ISSUE.md** - "Current Solution: String Interning with Bounded Cache"
- **STRING_INTERNING_IMPLEMENTATION.md** - "Current Implementation"
- **TIMELINE_PROTO_STRING_INTERNING.md** - "Phase 5 & 6: String Interning"

### Why String Interning Was Necessary
- **PROTO_STRING_INTERNING_ISSUE.md** - "Why String Interning Was Necessary"
- **INVESTIGATION_ANSWERS.md** - Complete explanation for Question 4
- **EXECUTIVE_SUMMARY.md** - "Why Other Solutions Don't Work"

### Memory and Performance Analysis
- **STRING_INTERNING_IMPLEMENTATION.md** - "Memory Behavior Analysis" and "Performance Considerations"
- **PROTO_STRING_INTERNING_ISSUE.md** - "Trade-offs" table

---

## Code References

### Main Implementation
- **File**: `datafusion/proto/src/physical_plan/from_proto.rs`
- **Lines**: 820-980 (cache structure, insertion, integration)
- **Key functions**:
  - `intern_format_str()` - Entry point for string interning
  - `format_options_from_proto()` - Deserialization entry point
  - `FormatStringCache::insert()` - Cache insertion logic
  - `FormatStringCache::get()` - Cache lookup logic

### Related Files
- `datafusion/proto/proto/datafusion.proto` - FormatOptions protobuf definition
- `datafusion/physical-expr-adapter/src/schema_rewriter.rs` - Uses cast options
- `datafusion/physical-expr/src/expressions/cast_column.rs` - CastColumnExpr definition

---

## Timeline Overview

```
June 27, 2025    → PhysicalExprAdapter introduced
Sept 23, 2025    → CastColumnExpr introduced
Oct 27, 2025     → Arrow upgraded to 57.0.0 (INTRODUCES LIFETIME REQUIREMENT)
Jan 23, 2026     → Format options serialized to protobuf (EXPOSES MISMATCH)
Jan 24, 2026     → String interning cache implemented (SOLVES MISMATCH)
```

**For detailed timeline**: See TIMELINE_PROTO_STRING_INTERNING.md

---

## FAQ Quick Links

### General Questions
- **"What is string interning?"**
  → PROTO_STRING_INTERNING_ISSUE.md - "Current Solution" section

- **"Why is this necessary?"**
  → EXECUTIVE_SUMMARY.md - "The Problem in One Picture" + "Why Other Solutions Don't Work"

- **"When was this added?"**
  → INVESTIGATION_ANSWERS.md - "Question 4: Was there a time..."

- **"Is this a memory leak?"**
  → PROTO_STRING_INTERNING_ISSUE.md - Trade-offs table; STRING_INTERNING_IMPLEMENTATION.md - "Memory Behavior Analysis"

### Implementation Questions
- **"How does the cache work?"**
  → STRING_INTERNING_IMPLEMENTATION.md - "Cache Structure" + "Insertion Logic"

- **"What's the cache limit?"**
  → STRING_INTERNING_IMPLEMENTATION.md - "Cache Size Limits"; STRING_INTERNING_QUICK_REFERENCE.md

- **"Is it thread-safe?"**
  → STRING_INTERNING_IMPLEMENTATION.md - "Thread Safety" section

- **"What error do I get if cache is full?"**
  → STRING_INTERNING_IMPLEMENTATION.md - "Error Handling" section

### Optimization Questions
- **"Can this be made faster?"**
  → STRING_INTERNING_IMPLEMENTATION.md - "Optimization Opportunities"

- **"Can we increase the cache limit?"**
  → EXECUTIVE_SUMMARY.md - "For Developers"

- **"How do I debug cache issues?"**
  → STRING_INTERNING_IMPLEMENTATION.md - "Observability & Debugging"

### Decision Questions
- **"Should we remove this?"**
  → EXECUTIVE_SUMMARY.md - "When This Can Be Fixed"

- **"What's the best approach going forward?"**
  → INVESTIGATION_ANSWERS.md - "Key Insights"

---

## Document Statistics

| Document | Lines | Read Time | Purpose |
|----------|-------|-----------|---------|
| EXECUTIVE_SUMMARY.md | ~170 | 5 min | High-level overview |
| INVESTIGATION_ANSWERS.md | ~320 | 10 min | Answer original questions |
| TIMELINE_PROTO_STRING_INTERNING.md | ~400 | 15 min | Historical evolution |
| STRING_INTERNING_IMPLEMENTATION.md | ~380 | 20 min | Technical details |
| STRING_INTERNING_QUICK_REFERENCE.md | ~130 | 3 min | Quick facts & lookup |
| PROTO_STRING_INTERNING_ISSUE.md | ~180 | 15 min | Problem statement |

**Total comprehensive reading time**: ~60 minutes (or select documents as needed)

---

## How to Navigate This Documentation

### If you have 5 minutes
→ Read **EXECUTIVE_SUMMARY.md**

### If you have 15 minutes
→ Read **EXECUTIVE_SUMMARY.md** + **STRING_INTERNING_QUICK_REFERENCE.md**

### If you have 30 minutes
→ Read **EXECUTIVE_SUMMARY.md** + **INVESTIGATION_ANSWERS.md** + **STRING_INTERNING_QUICK_REFERENCE.md**

### If you have 60 minutes
→ Read all documents in this order:
1. EXECUTIVE_SUMMARY.md
2. STRING_INTERNING_QUICK_REFERENCE.md
3. INVESTIGATION_ANSWERS.md
4. TIMELINE_PROTO_STRING_INTERNING.md
5. STRING_INTERNING_IMPLEMENTATION.md
6. PROTO_STRING_INTERNING_ISSUE.md (reference as needed)

### If you need to understand specific topics
→ Use the "Key Topics by Document" section above to jump to relevant sections

---

## Summary

This investigation documents a **9-month evolution** from the introduction of `PhysicalExprAdapter` (June 2025) to the implementation of string interning (January 2026). The investigation reveals that string interning is not a design preference but a necessary solution to an unavoidable collision between Arrow's API requirements and protobuf's data model.

All documentation has been created to provide multiple entry points for different audiences and use cases, from executive overview to deep technical implementation details.
