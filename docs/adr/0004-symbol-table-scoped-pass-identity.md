# ADR-0004: Pass identity is table-agnostic text

Status: Accepted

## Context
The original Pass interface returned a SymbolId bound to whichever
SymbolTable constructed the pass. Registering the (static) passes with a
second table produced dangling pool lookups — caught by the test suites
when two PassContexts coexisted.

## Decision
Pass identity is carried as stable text (Pass::nameText()); SymbolIds are
resolved per-context by the registry and pass consumers
(PassRegistry::add interns into the caller's table). SymbolTables are
process-scoped resources; ids are never compared across tables.

## Consequences
Multiple SymbolTables coexist safely (tools, tests, concurrent compile
contexts); registry contracts are always consistent with the querying
table.
