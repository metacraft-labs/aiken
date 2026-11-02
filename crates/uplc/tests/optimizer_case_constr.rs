//! Parse valid PlutusV3 programs and run the public optimiser with the real CEK
//! evaluator, without mocks. Compare values, evaluation errors, and trace logs
//! before optimisation, after one pass, and after a second pass. One test
//! also applies a context to an optimiser-generated Case/Constr before optimising
//! again, recreating the consumer's partial-evaluation path.
use uplc::{
    ast::{Name, NamedDeBruijn, Program, Term},
    machine::{cost_model::ExBudget, eval_result::EvalResult},
    optimize::aiken_optimize_and_intern,
    parser,
};

fn evaluate(program: Program<Name>) -> EvalResult {
    let program: Program<NamedDeBruijn> = program.try_into().unwrap();
    program.eval(ExBudget::default())
}

fn preserves_evaluation(source: &str, expected: Term<NamedDeBruijn>, logs: &[&str]) {
    let program = parser::program(source).unwrap();
    let before = evaluate(program.clone());
    assert_eq!(before.result, Ok(expected));
    assert_eq!(before.logs(), logs);

    let optimized = aiken_optimize_and_intern(program);
    let after = evaluate(optimized.clone());
    assert_eq!(after.result, before.result);
    assert_eq!(after.logs(), before.logs());

    let twice = evaluate(aiken_optimize_and_intern(optimized));
    assert_eq!(twice.result, before.result);
    assert_eq!(twice.logs(), before.logs());
}

#[test]
fn constructor_fields_are_counted_and_substituted() {
    preserves_evaluation(
        r#"(program 1.1.0 [(lam x (constr 0 x (constr 1 x))) (con string "field")])"#,
        Term::constr(
            0,
            vec![
                Term::string("field"),
                Term::constr(1, vec![Term::string("field")]),
            ],
        ),
        &[],
    );
}

#[test]
fn case_scrutinee_and_all_branches_are_substituted() {
    for tag in [0, 1] {
        preserves_evaluation(
            &format!(
                r#"(program 1.1.0
                  [(lam x (case (constr {tag} x) (lam y x) (lam z z))) (con string "branch")])"#
            ),
            Term::string("branch"),
            &[],
        );
    }
    preserves_evaluation(
        "(program 1.1.0 [(lam c (case c (con integer 42))) (constr 0)])",
        Term::integer(42.into()),
        &[],
    );
}

#[test]
fn identity_uses_inside_case_and_constructor_are_replaced() {
    preserves_evaluation(
        "(program 1.1.0
          [(lam id (case (constr 0 [id (con integer 20)])
            (lam n (constr 1 [id n] [id (con integer 22)])))) (lam a a)])",
        Term::constr(1, vec![Term::integer(20.into()), Term::integer(22.into())]),
        &[],
    );
}

#[test]
fn case_branch_does_not_hide_an_eager_error() {
    let program =
        parser::program("(program 1.1.0 [(lam x (case (constr 0) (con integer 7) x)) (error)])")
            .unwrap();
    let before = evaluate(program.clone());
    assert!(before.result.is_err());
    let after = evaluate(aiken_optimize_and_intern(program));
    assert_eq!(after.result, before.result);
    assert_eq!(after.logs(), before.logs());
}

#[test]
fn case_branch_does_not_hide_an_eager_trace() {
    preserves_evaluation(
        r#"(program 1.1.0
          [(lam x (case (constr 0) (con integer 7) x))
            [(force (builtin trace)) (con string "eager") (con integer 9)]])"#,
        Term::integer(7.into()),
        &["eager"],
    );
}

#[test]
fn field_lambdas_do_not_consume_surrounding_case_arguments() {
    preserves_evaluation(
        "(program 1.1.0
          [(lam x [(case (constr 0 (lam ignored x)) (lam f f)) (con integer 0)])
            [(builtin addInteger) (con integer 20) (con integer 22)]])",
        Term::integer(42.into()),
        &[],
    );
}

#[test]
fn optimize_again_after_applying_context_to_generated_case() {
    // The final optimiser pass packs three arguments into Case/Constr. The
    // consumer then applies a constant context and optimises that result again.
    let program = parser::program(
        "(program 1.1.0 (lam context [context (con integer 1) (con integer 2) (con integer 3)]))",
    )
    .unwrap();
    let optimized = aiken_optimize_and_intern(program);
    let Term::Lambda { body, .. } = &optimized.term else {
        panic!("expected a validator awaiting its context");
    };
    assert!(matches!(body.as_ref(), Term::Case { .. }));
    let context =
        parser::term("(lam a (lam b (lam c [(builtin addInteger) a [(builtin addInteger) b c]])))")
            .unwrap();
    let applied = optimized.apply_term(&context);
    let before = evaluate(applied.clone());
    assert_eq!(before.result, Ok(Term::integer(6.into())));
    let after = evaluate(aiken_optimize_and_intern(applied));
    assert_eq!(after.result, before.result);
    assert_eq!(after.logs(), before.logs());
}

#[test]
fn currying_keeps_constructor_fields_in_distinct_scopes() {
    let program = parser::program(
        "(program 1.1.0 (lam x (constr 0
          [(builtin addInteger) (con integer 1) x]
          [(builtin addInteger) (con integer 1) x]
          [(builtin addInteger) (con integer 1) x])))",
    )
    .unwrap();
    let argument = Term::integer(41.into());
    let before = evaluate(program.clone().apply_term(&argument));
    assert_eq!(
        before.result,
        Ok(Term::constr(0, vec![Term::integer(42.into()); 3]))
    );
    let optimized = aiken_optimize_and_intern(program);
    let after = evaluate(optimized.clone().apply_term(&argument));
    assert_eq!(after.result, before.result);
    let twice = evaluate(aiken_optimize_and_intern(optimized).apply_term(&argument));
    assert_eq!(twice.result, before.result);
}

#[test]
fn currying_keeps_case_branches_in_distinct_scopes() {
    for tag in [0, 1, 2] {
        let program = parser::program(&format!(
            "(program 1.1.0 (lam x (case (constr {tag})
              [(builtin addInteger) (con integer 1) x]
              [(builtin addInteger) (con integer 1) x]
              [(builtin addInteger) (con integer 1) x])))"
        ))
        .unwrap();
        let argument = Term::integer(41.into());
        let before = evaluate(program.clone().apply_term(&argument));
        assert_eq!(before.result, Ok(Term::integer(42.into())));
        let optimized = aiken_optimize_and_intern(program);
        let after = evaluate(optimized.clone().apply_term(&argument));
        assert_eq!(after.result, before.result);
        let twice = evaluate(aiken_optimize_and_intern(optimized).apply_term(&argument));
        assert_eq!(twice.result, before.result);
    }
}

#[test]
fn currying_does_not_hoist_work_out_of_untaken_case_branches() {
    let program = parser::program(
        r#"(program 1.1.0 (lam x (case (constr 0)
          (con integer 7)
          [(builtin subtractInteger) [(force (builtin trace)) (con string "untaken") x] x]
          [(builtin subtractInteger) [(force (builtin trace)) (con string "untaken") x] x]
          [(builtin subtractInteger) [(force (builtin trace)) (con string "untaken") x] x])))"#,
    )
    .unwrap();
    let argument = Term::integer(41.into());
    let before = evaluate(program.clone().apply_term(&argument));
    assert_eq!(before.result, Ok(Term::integer(7.into())));
    assert!(before.logs().is_empty());
    let optimized = aiken_optimize_and_intern(program);
    let after = evaluate(optimized.clone().apply_term(&argument));
    assert_eq!(after.result, before.result);
    assert_eq!(after.logs(), before.logs());
    let twice = evaluate(aiken_optimize_and_intern(optimized).apply_term(&argument));
    assert_eq!(twice.result, before.result);
    assert_eq!(twice.logs(), before.logs());
}

#[test]
fn currying_preserves_each_constructor_fields_trace() {
    let program = parser::program(
        r#"(program 1.1.0 (lam x (constr 0
          [(builtin subtractInteger) [(force (builtin trace)) (con string "field") x] x]
          [(builtin subtractInteger) [(force (builtin trace)) (con string "field") x] x]
          [(builtin subtractInteger) [(force (builtin trace)) (con string "field") x] x])))"#,
    )
    .unwrap();
    let argument = Term::integer(41.into());
    let before = evaluate(program.clone().apply_term(&argument));
    assert_eq!(
        before.result,
        Ok(Term::constr(0, vec![Term::integer(0.into()); 3]))
    );
    assert_eq!(before.logs(), ["field", "field", "field"]);
    let optimized = aiken_optimize_and_intern(program);
    let after = evaluate(optimized.clone().apply_term(&argument));
    assert_eq!(after.result, before.result);
    assert_eq!(after.logs(), before.logs());
    let twice = evaluate(aiken_optimize_and_intern(optimized).apply_term(&argument));
    assert_eq!(twice.result, before.result);
    assert_eq!(twice.logs(), before.logs());
}

#[test]
fn currying_does_not_hoist_errors_out_of_untaken_case_branches() {
    let program = parser::program(
        "(program 1.1.0 (lam x (case (constr 0)
          (con integer 7)
          [(builtin subtractInteger) (error) x]
          [(builtin subtractInteger) (error) x]
          [(builtin subtractInteger) (error) x])))",
    )
    .unwrap();
    let argument = Term::integer(41.into());
    let before = evaluate(program.clone().apply_term(&argument));
    assert_eq!(before.result, Ok(Term::integer(7.into())));
    let optimized = aiken_optimize_and_intern(program);
    let after = evaluate(optimized.clone().apply_term(&argument));
    assert_eq!(after.result, before.result);
    let twice = evaluate(aiken_optimize_and_intern(optimized).apply_term(&argument));
    assert_eq!(twice.result, before.result);
}
