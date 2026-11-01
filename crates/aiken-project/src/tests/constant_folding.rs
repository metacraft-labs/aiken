//! What the compiler's constant folding leaves in a generated program.
//!
//! A consumer that steps the generated UPLC, such as a time-travel debugger,
//! can only show what is in the program. These tests pin down how much of a
//! source computation survives code generation, so that a consumer's statement
//! of that limit fails here if the compiler's behaviour changes:
//!
//! - a nullary `test` over constants compiles to its answer, at every tracing
//!   level, unless the module has a `trace` for the level to keep;
//! - a validator keeps a computation over its runtime input. A `let` used once
//!   is inlined and its name goes with it; one used more than once stays as a
//!   named parameter of the compiler's program, and serialisation erases that
//!   name, so the stored script has no source names at all;
//! - a validator that computes only over constants keeps its wrapper around the
//!   answer and nothing of the computation.
//!
//! No mocks: every stage is the compiler's own, run through `TestProject`.

use super::TestProject;
use crate::module::CheckedModules;
use aiken_lang::ast::{Definition, TraceLevel, Tracing};
use std::rc::Rc;
use uplc::ast::{Constant, DeBruijn, Name, Program, Term};

/// One `test`, with a computation spelled out across named intermediates.
const FLOW_TEST: &str = r#"
    fn compute() -> Int {
      let a = 10
      let b = 32
      let sum_val = a + b
      let doubled = sum_val * 2
      let final_result = doubled + a
      final_result
    }

    test flow_test() {
      compute() == 94
    }
"#;

/// The same computation with a `trace` between each binding.
const TRACING_TEST: &str = r#"
    fn compute() -> Int {
      let a = 4
      trace @"after-a": a
      let b = 5
      trace @"after-b": b
      let sum_val = a + b
      trace @"final-sum": sum_val
      sum_val
    }

    test tracing() {
      compute() == 9
    }
"#;

/// The same computation over a value only known when the script runs.
const VALIDATOR_OVER_INPUT: &str = r#"
    validator flow {
      mint(redeemer: Int, _policy_id: ByteArray, _transaction: Data) {
        let a = redeemer
        let b = 32
        let sum_val = a + b
        let doubled = sum_val * 2
        let final_result = doubled + a
        final_result == 94
      }
    }
"#;

/// A computation over a runtime input whose bindings are each used twice, so
/// the optimiser has a reason to keep them.
const VALIDATOR_REUSING_ITS_BINDINGS: &str = r#"
    validator flow {
      mint(redeemer: Int, _policy_id: ByteArray, _transaction: Data) {
        let sum_val = redeemer + 32
        let doubled = sum_val * sum_val
        let final_result = doubled + sum_val + doubled
        final_result + final_result == 94
      }
    }
"#;

/// The same computation in a validator that reads none of its inputs.
const VALIDATOR_OVER_CONSTANTS: &str = r#"
    fn compute() -> Int {
      let a = 10
      let b = 32
      let sum_val = a + b
      let doubled = sum_val * 2
      let final_result = doubled + a
      final_result
    }

    validator flow {
      mint(_redeemer: Data, _policy_id: ByteArray, _transaction: Data) {
        compute() == 94
      }
    }
"#;

const TRACE_LEVELS: [TraceLevel; 3] =
    [TraceLevel::Silent, TraceLevel::Compact, TraceLevel::Verbose];

/// Run the compiler's route for the one `test` or validator in `source_code`.
///
/// `tracing` goes to the type checker and to the code generator alike, because
/// the type checker is what erases a `trace`.
fn generate(source_code: &str, level: TraceLevel) -> Program<Name> {
    let tracing = Tracing::All(level);
    let mut project = TestProject::new();
    let parsed = project.parse(source_code);
    let modules = CheckedModules::singleton(project.check_with_tracing(parsed, tracing));
    let mut generator = project.new_generator(tracing);
    let module = modules.values().next().expect("one module was checked");

    module
        .ast
        .definitions()
        .find_map(|definition| match definition {
            Definition::Test(test) => Some(generator.generate_raw(&test.body, &[], &module.name)),
            Definition::Validator(validator) => Some(generator.generate(validator, &module.name)),
            _ => None,
        })
        .expect("the source declares a test or a validator")
}

fn bool_constant(value: bool) -> Term<Name> {
    Term::Constant(Rc::new(Constant::Bool(value)))
}

#[test]
fn a_nullary_test_over_constants_compiles_to_its_answer() {
    assert_eq!(
        generate(FLOW_TEST, TraceLevel::Silent).term,
        bool_constant(true)
    );
}

/// The control for the test above: the constant is the computed answer, and not
/// a constant the compiler emits whatever the source says.
#[test]
fn the_answer_follows_the_source() {
    assert_eq!(
        generate(&FLOW_TEST.replace("94", "95"), TraceLevel::Silent).term,
        bool_constant(false)
    );
}

/// Tracing only keeps what the source traces. It is no way to keep a
/// computation: a module with no `trace` compiles identically at every level.
#[test]
fn no_tracing_level_keeps_a_computation_the_source_does_not_trace() {
    for level in TRACE_LEVELS {
        assert_eq!(
            generate(FLOW_TEST, level).term,
            bool_constant(true),
            "at {level:?}"
        );
    }
}

/// A `trace` is an effect, so it stops the folding where it is kept and goes
/// with the rest where it is erased.
#[test]
fn a_kept_trace_is_what_stops_a_nullary_test_folding() {
    assert_eq!(
        generate(TRACING_TEST, TraceLevel::Silent).term,
        bool_constant(true)
    );

    for level in [TraceLevel::Compact, TraceLevel::Verbose] {
        let program = generate(TRACING_TEST, level);
        assert!(
            !matches!(program.term, Term::Constant(_)),
            "at {level:?}: {}",
            program.to_pretty()
        );
    }
}

/// A computation over a runtime input stays in the program, as arithmetic. The
/// bindings here are each used once, so the optimiser inlines them and their
/// names are gone before anything is serialised.
#[test]
fn a_validator_keeps_a_computation_over_its_input_and_inlines_single_use_names() {
    let program = generate(VALIDATOR_OVER_INPUT, TraceLevel::Silent).to_pretty();

    for builtin in ["addInteger", "multiplyInteger", "equalsInteger"] {
        assert!(program.contains(builtin), "{builtin} in {program}");
    }
    for name in ["sum_val", "doubled", "final_result"] {
        assert!(!program.contains(name), "{name} in {program}");
    }
}

/// A binding the optimiser keeps does keep its name in the compiler's program,
/// so the absence above is not the whole story. It does not reach the chain:
/// the stored form is De Bruijn indexed, and decoding it gives back indices and
/// no names. A stepping consumer works from the stored form, so it has no way
/// to read `sum_val` off the program whether or not the binding was inlined.
#[test]
fn serialisation_erases_the_name_of_a_binding_the_optimiser_keeps() {
    let names = ["sum_val", "doubled", "final_result"];

    let compiled = generate(VALIDATOR_REUSING_ITS_BINDINGS, TraceLevel::Silent);
    let pretty = compiled.to_pretty();
    for name in names {
        assert!(pretty.contains(name), "{name} in {pretty}");
    }

    let stored = compiled
        .to_debruijn()
        .expect("the compiler's program has no free variables")
        .to_flat()
        .expect("the compiler's program serialises");
    let decoded = Program::<DeBruijn>::from_flat(&stored)
        .expect("the stored program decodes")
        .to_pretty();

    // The control: the decoded text is the same program, and not one that has
    // lost its contents along with the names.
    assert!(decoded.contains("multiplyInteger"), "{decoded}");
    for name in names {
        assert!(!decoded.contains(name), "{name} in {decoded}");
    }
}

/// A validator is never a bare constant, because it has to be applied to its
/// script context, but its body folds as a `test`'s does: what is left is the
/// wrapper around the answer.
#[test]
fn a_validator_over_constants_keeps_its_wrapper_and_folds_its_body() {
    let program = generate(VALIDATOR_OVER_CONSTANTS, TraceLevel::Silent);
    assert!(!matches!(program.term, Term::Constant(_)));

    let program = program.to_pretty();
    assert!(program.contains("(con bool True)"), "{program}");
    for builtin in ["addInteger", "multiplyInteger"] {
        assert!(!program.contains(builtin), "{builtin} in {program}");
    }
}
