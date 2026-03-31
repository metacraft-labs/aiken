/// Tests for the public stepping API (get_initial_machine_state + step).
///
/// These tests verify that:
/// 1. Step-by-step execution produces the same result as Machine::run()
/// 2. get_initial_machine_state() returns a Compute state
/// 3. Stepping a Done state returns Done
/// 4. Budget tracking across steps matches run() budget
use pallas_primitives::conway::Language;
use uplc::{
    ast::{DeBruijn, NamedDeBruijn, Program, Term},
    machine::{
        cost_model::{CostModel, ExBudget},
        Machine, MachineState,
    },
    parser,
};

/// Helper: parse a UPLC program from text and convert to NamedDeBruijn.
fn parse_program(code: &str) -> Program<NamedDeBruijn> {
    let parsed = parser::program(code).unwrap();
    let debruijn: Program<DeBruijn> = parsed.try_into().unwrap();
    let named: Program<NamedDeBruijn> = debruijn.try_into().unwrap();
    named
}

/// Helper: run a program to completion using step(), returning the
/// final term and remaining budget.
fn step_to_completion(
    machine: &mut Machine,
    term: Term<NamedDeBruijn>,
) -> Result<Term<NamedDeBruijn>, uplc::machine::Error> {
    let mut state = machine.get_initial_machine_state(term)?;
    loop {
        match state {
            MachineState::Done(t) => return Ok(t),
            _ => {
                state = machine.step(state)?;
            }
        }
    }
}

/// Verify that step-by-step execution of the identity function applied to
/// an integer produces the same result as Machine::run().
#[test]
fn step_matches_run_identity() {
    // (lam x x) 42  =>  42
    let code = "(program 1.0.0 [(lam i_0 i_0) (con integer 42)])";
    let program = parse_program(code);
    let budget = ExBudget::default();

    // Run via run()
    let mut machine_run =
        Machine::new(Language::PlutusV3, CostModel::default(), budget, 200);
    let run_result = machine_run.run(program.term.clone()).unwrap();

    // Run via step()
    let mut machine_step =
        Machine::new(Language::PlutusV3, CostModel::default(), budget, 200);
    let step_result = step_to_completion(&mut machine_step, program.term).unwrap();

    assert_eq!(
        format!("{:?}", run_result),
        format!("{:?}", step_result),
        "step-by-step and run() must produce the same result"
    );
}

/// Verify budget consumption matches between run() and step().
#[test]
fn step_budget_matches_run() {
    let code = "(program 1.0.0 [(lam i_0 i_0) (con integer 42)])";
    let program = parse_program(code);
    let budget = ExBudget::default();

    let mut machine_run =
        Machine::new(Language::PlutusV3, CostModel::default(), budget, 200);
    let _ = machine_run.run(program.term.clone()).unwrap();

    let mut machine_step =
        Machine::new(Language::PlutusV3, CostModel::default(), budget, 200);
    let _ = step_to_completion(&mut machine_step, program.term).unwrap();

    assert_eq!(
        machine_run.ex_budget.cpu, machine_step.ex_budget.cpu,
        "CPU budget must match between run() and step()"
    );
    assert_eq!(
        machine_run.ex_budget.mem, machine_step.ex_budget.mem,
        "Memory budget must match between run() and step()"
    );
}

/// Verify that get_initial_machine_state() returns a Compute state.
#[test]
fn initial_state_is_compute() {
    let code = "(program 1.0.0 (con integer 1))";
    let program = parse_program(code);
    let budget = ExBudget::default();

    let mut machine =
        Machine::new(Language::PlutusV3, CostModel::default(), budget, 200);
    let state = machine.get_initial_machine_state(program.term).unwrap();

    assert!(
        matches!(state, MachineState::Compute(..)),
        "initial state must be Compute"
    );
}

/// Verify that stepping a Done state returns Done.
#[test]
fn step_done_returns_done() {
    let budget = ExBudget::default();
    let mut machine =
        Machine::new(Language::PlutusV3, CostModel::default(), budget, 200);

    let done_term = Term::integer(99.into());
    let state = MachineState::Done(done_term.clone());

    let result = machine.step(state).unwrap();
    match result {
        MachineState::Done(t) => {
            assert_eq!(
                format!("{:?}", t),
                format!("{:?}", done_term),
                "Done state must preserve the term"
            );
        }
        _ => panic!("stepping Done must return Done"),
    }
}

/// Test step-by-step execution with a builtin (addInteger).
#[test]
fn step_matches_run_arithmetic() {
    let code = "(program 1.0.0
      [(lam i_0 [[(builtin addInteger) (con integer 10)] i_0]) (con integer 32)]
    )";
    let program = parse_program(code);
    let budget = ExBudget::default();

    let mut machine_run =
        Machine::new(Language::PlutusV3, CostModel::default(), budget, 200);
    let run_result = machine_run.run(program.term.clone()).unwrap();

    let mut machine_step =
        Machine::new(Language::PlutusV3, CostModel::default(), budget, 200);
    let step_result = step_to_completion(&mut machine_step, program.term).unwrap();

    assert_eq!(format!("{:?}", run_result), format!("{:?}", step_result));
    assert_eq!(machine_run.ex_budget.cpu, machine_step.ex_budget.cpu);
    assert_eq!(machine_run.ex_budget.mem, machine_step.ex_budget.mem);
}

/// Test that trace logs are collected identically between run() and step().
#[test]
fn step_traces_match_run() {
    // trace is a forced builtin in Plutus
    let code = "(program 1.0.0
      [[(force (builtin trace)) (con string \"hello\")] (con integer 1)]
    )";
    let program = parse_program(code);
    let budget = ExBudget::default();

    let mut machine_run =
        Machine::new(Language::PlutusV3, CostModel::default(), budget, 200);
    let _ = machine_run.run(program.term.clone()).unwrap();

    let mut machine_step =
        Machine::new(Language::PlutusV3, CostModel::default(), budget, 200);
    let _ = step_to_completion(&mut machine_step, program.term).unwrap();

    let run_traces: Vec<String> = machine_run
        .traces
        .iter()
        .map(|t| format!("{}", t))
        .collect();
    let step_traces: Vec<String> = machine_step
        .traces
        .iter()
        .map(|t| format!("{}", t))
        .collect();

    assert_eq!(
        run_traces, step_traces,
        "traces must match between run() and step()"
    );
}

/// Test step-by-step with delay/force (common in Plutus programs).
#[test]
fn step_matches_run_delay_force() {
    let code = "(program 1.0.0
      (force (delay (con integer 7)))
    )";
    let program = parse_program(code);
    let budget = ExBudget::default();

    let mut machine_run =
        Machine::new(Language::PlutusV3, CostModel::default(), budget, 200);
    let run_result = machine_run.run(program.term.clone()).unwrap();

    let mut machine_step =
        Machine::new(Language::PlutusV3, CostModel::default(), budget, 200);
    let step_result = step_to_completion(&mut machine_step, program.term).unwrap();

    assert_eq!(format!("{:?}", run_result), format!("{:?}", step_result));
    assert_eq!(machine_run.ex_budget.cpu, machine_step.ex_budget.cpu);
}

/// Count the number of steps taken and verify it's > 0.
#[test]
fn step_count_is_nonzero() {
    let code = "(program 1.0.0 [(lam i_0 i_0) (con integer 42)])";
    let program = parse_program(code);
    let budget = ExBudget::default();

    let mut machine =
        Machine::new(Language::PlutusV3, CostModel::default(), budget, 200);
    let mut state = machine.get_initial_machine_state(program.term).unwrap();
    let mut count = 0u64;
    loop {
        match state {
            MachineState::Done(_) => break,
            _ => {
                state = machine.step(state).unwrap();
                count += 1;
            }
        }
    }
    assert!(count > 0, "a non-trivial program must take at least one step");
}

/// Test with constr/case (PlutusV3 feature).
#[test]
fn step_matches_run_constr_case() {
    // constr 1 builds a constructor with tag=1 and two fields
    // case dispatches on the tag: branch 0 returns first arg, branch 1 returns second
    let code = "(program 1.0.0
      (case
        (constr 1 (con integer 10) (con integer 20))
        (lam i_0 (lam i_1 i_0))
        (lam i_0 (lam i_1 i_1))
      )
    )";
    let program = parse_program(code);
    let budget = ExBudget::default();

    let mut machine_run =
        Machine::new(Language::PlutusV3, CostModel::default(), budget, 200);
    let run_result = machine_run.run(program.term.clone()).unwrap();

    let mut machine_step =
        Machine::new(Language::PlutusV3, CostModel::default(), budget, 200);
    let step_result = step_to_completion(&mut machine_step, program.term).unwrap();

    assert_eq!(format!("{:?}", run_result), format!("{:?}", step_result));
    assert_eq!(machine_run.ex_budget.cpu, machine_step.ex_budget.cpu);
    assert_eq!(machine_run.ex_budget.mem, machine_step.ex_budget.mem);
}
