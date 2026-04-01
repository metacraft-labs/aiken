# Upstream PR: Public Stepping API for UPLC CEK Machine

**Target:** aiken-lang/aiken
**Branch:** codetracer-stepping-api
**Scope:** 2 files, +26/-3 lines

## Summary

Add a public stepping API to the UPLC CEK machine, enabling external tools to drive evaluation one transition at a time without forking the crate.

Two new public methods on `Machine`:

- `get_initial_machine_state(term) -> Result<MachineState, Error>` — returns the initial state, spending the startup budget
- `step(state) -> Result<MachineState, Error>` — performs a single CEK transition

Three visibility changes (all previously private/pub(super)):

- `MachineState` enum — made `pub`
- `Context` enum — made `pub`
- `value::Env` type alias — made `pub`, re-exported as `MachineEnv`

## Motivation

Tools like time-travel debuggers, profilers, and execution tracers need to observe every step of UPLC evaluation. Currently, the only way to do this is to fork the `uplc` crate, since `Machine::run()` executes to completion without any callback or observer mechanism.

This change enables a step-by-step execution pattern:

```rust
let mut state = machine.get_initial_machine_state(term)?;
loop {
    match state {
        MachineState::Done(t) => break Ok(t),
        _ => state = machine.step(state)?,
    }
}
```

Concrete consumers include [CodeTracer](https://github.com/metacraft-labs/codetracer) (time-travel debugger for smart contracts) and [Gastronomy](https://github.com/SundaeSwap-finance/gastronomy) (Cardano transaction debugger).

## Relationship to PR #1250

PR #1250 (SundaeSwap) includes a superset of these changes as part of a larger source map threading effort. This PR is the minimal, independently reviewable subset — it focuses solely on making the execution steppable without the source map complexity. If #1250 merges first, this PR can be closed.

## Design

- **Purely additive**: No existing APIs are modified or removed. `Machine::run()` is unchanged.
- **Zero behavioral change**: `step()` delegates directly to the existing private `compute()` and `return_compute()` methods. Step-by-step execution produces identical results and budget consumption as `run()`.
- **Zero performance impact**: Making types public and adding methods has no runtime cost. Benchmark execution units are identical.

## Testing

- **828 workspace tests pass** (477 aiken-lang, 177 aiken-project, 143 uplc, 9 stepping, etc.)
- **126 acceptance tests pass** (script_context tests excluded — require cbor-diag tool)
- **All benchmarks pass** with identical execution units
- **Clippy clean**, no warnings
- **9 dedicated stepping tests** verify:
  - Run/step equivalence for identity, arithmetic, delay/force, constr/case programs
  - Budget consumption (CPU + memory) matches between run() and step()
  - Trace log collection is identical
  - Initial state is Compute, Done state is terminal
