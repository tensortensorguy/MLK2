// Math pass unit tests (Rules 10, 22, 33, 87): idempotence, legality gates,
// canonicalization correctness.
#include <cmath>

#include "mlk/core/diagnostics.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_builder.h"
#include "mlk/ir/graph_hash.h"
#include "mlk/ir/graph_printer.h"
#include "mlk/pass/register_all.h"
#include "mlk/pass/pass_registry.h"
#include "mlk/runtime/execution.h"
#include "mlk/type/domain_profile.h"
#include "mlk/verifier/graph_verifier.h"

#include "mlk_test.h"

namespace {
struct PassEnv {
    mlk::SymbolTable symbols;
    mlk::DiagnosticEngine diag;
    mlk::MathDomainProfile profile;
    mlk::AccuracyContract contract;
    mlk::PassContext ctx;

    PassEnv(bool allowReassoc = false) {
        mlk::passes::registerAllPasses(symbols);
        profile.name = symbols.intern("test");
        profile.capabilities.set(mlk::Capability::HasNumericValues);
        profile.capabilities.set(mlk::Capability::HasFloatingPoint);
        profile.capabilities.set(mlk::Capability::HasTensorDomain);
        profile.capabilities.set(mlk::Capability::HasKernelFusion);
        profile.capabilities.set(mlk::Capability::HasAlgebraicRewriting);
        profile.capabilities.set(mlk::Capability::HasDerivatives);
        profile.capabilities.set(mlk::Capability::HasApproximation);
        profile.lawCommutativeMul = mlk::TriState::True;
        profile.lawAssociativeAdd =
            allowReassoc ? mlk::TriState::True : mlk::TriState::Unknown;
        profile.numeric.allowReassociation = allowReassoc;
        ctx.domainProfile = &profile;
        ctx.accuracy = &contract;
        ctx.diag = &diag;
        ctx.symbols = &symbols;
        ctx.tier = mlk::Tier::Tier1;
    }

    mlk::Pass* pass(const char* name) {
        return mlk::PassRegistry::instance().byName(symbols, symbols.intern(name));
    }
};
}  // namespace

MLK_TEST(math, constant_fold_arithmetic) {
    PassEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    auto two = b.constant(2.0, f64);
    auto three = b.constant(3.0, f64);
    auto sum = b.op(mlk::MathOp::Add, {two, three});
    MLK_CHECK(sum.has_value());
    b.output(*sum);
    auto pass = env.pass("math.constant_fold");
    MLK_CHECK(pass != nullptr);
    auto r = pass->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(r->changed);
    // 2+3 folded to constant 5; the original add node is dead (Rule 21).
    const mlk::Value& v = b.graph().value(b.graph().representative(*sum));
    MLK_CHECK(v.kind == mlk::ValueKind::Constant);
    MLK_CHECK_NEAR(v.constant.f64, 5.0, 1e-12);
}

MLK_TEST(math, constant_fold_never_creates_nan) {
    // Rule 87: log(-1) is a domain violation; no fold, no NaN creation.
    PassEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    auto neg = b.constant(-1.0, f64);
    auto l = b.op(mlk::MathOp::Log, {neg});
    MLK_CHECK(l.has_value());
    b.output(*l);
    auto pass = env.pass("math.constant_fold");
    auto r = pass->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(!r->changed);
    const mlk::Value& v = b.graph().value(b.graph().representative(*l));
    MLK_CHECK(v.kind == mlk::ValueKind::NodeResult);  // unfolded
}

MLK_TEST(math, identity_elim_x_plus_zero_domain_gated) {
    // x + 0 -> x is exact for every x EXCEPT x = -0.0 (IEEE: (-0)+(+0)=+0),
    // so under the default sign-preserving profile it must NOT fire (see
    // identity_elim_neg_zero_gate_is_operand_signed). When the domain
    // drops negative zero, the fold is legal.
    PassEnv env;
    env.profile.numeric.preserveNegativeZero = false;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto zero = b.constant(0.0, f64);
    auto sum = b.op(mlk::MathOp::Add, {x, zero});
    MLK_CHECK(sum.has_value());
    b.output(*sum);
    // property.infer first (identity elim requires facts)
    MLK_CHECK(env.pass("property.infer")->run(env.ctx, b.graph()).has_value());
    auto pass = env.pass("math.identity_elim");
    auto r = pass->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(r->changed);
    MLK_CHECK_EQ(b.graph().representative(*sum), x);
}

MLK_TEST(math, identity_elim_neg_zero_gate_is_operand_signed) {
    // Rule 90, corrected semantics (regression for the inverted gate):
    //   x + (-0.0) -> x is EXACT for every x — IEEE round-to-nearest gives
    //   (+0)+(-0)=+0 and (-0)+(-0)=-0, so the result always equals x. It
    //   folds under sign preservation.
    //   x + (+0.0) -> x is WRONG for x = -0.0 (IEEE gives +0): it folds
    //   only when the domain drops negative zero.
    PassEnv env;  // default profile: preserveNegativeZero = true
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto nz = b.constant(-0.0, f64);
    auto sum = b.op(mlk::MathOp::Add, {x, nz});
    MLK_CHECK(sum.has_value());
    b.output(*sum);
    MLK_CHECK(env.pass("property.infer")->run(env.ctx, b.graph()).has_value());
    auto r = env.pass("math.identity_elim")->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(r->changed);  // x + (-0.0) -> x is exact: eliminated
    MLK_CHECK_EQ(b.graph().representative(*sum), x);

    // The unsound form stays: add(x, +0.0) under sign preservation.
    mlk::GraphBuilder b2(env.symbols);
    const mlk::ValueId y = b2.placeholder("x", f64);
    auto pz = b2.constant(0.0, f64);
    auto sum2 = b2.op(mlk::MathOp::Add, {y, pz});
    MLK_CHECK(sum2.has_value());
    b2.output(*sum2);
    MLK_CHECK(env.pass("property.infer")->run(env.ctx, b2.graph()).has_value());
    auto r2 = env.pass("math.identity_elim")->run(env.ctx, b2.graph());
    MLK_CHECK(r2.has_value());
    MLK_CHECK(!r2->changed);  // NOT eliminated: (-0) + (+0) = +0 != x
    MLK_CHECK_EQ(b2.graph().representative(*sum2), *sum2);

    // When the domain drops -0, add(x, +0.0) folds.
    env.profile.numeric.preserveNegativeZero = false;
    auto r3 = env.pass("math.identity_elim")->run(env.ctx, b2.graph());
    MLK_CHECK(r3.has_value());
    MLK_CHECK(r3->changed);
    MLK_CHECK_EQ(b2.graph().representative(*sum2), y);
}

MLK_TEST(math, identity_elim_int_constants_typed_probes) {
    // Regression (certified miscompile): int constants probed through the
    // FP slot of isConst() read as 0.0, folding add(x, 5) -> x. The typed
    // probes keep every non-zero int constant intact while the legal
    // int identities (x+0, x*1, x*0 with int operands) fire.
    PassEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType i64t =
        mlk::MathType::scalar(mlk::Domain::Int, mlk::Dtype::I64);
    const mlk::ValueId x = b.placeholder("x", i64t);
    auto five = b.integer(5, i64t);
    auto sum = b.op(mlk::MathOp::Add, {x, five});
    MLK_CHECK(sum.has_value());
    auto timesOne = b.op(mlk::MathOp::Mul, {x, b.integer(1, i64t)});
    MLK_CHECK(timesOne.has_value());
    auto plusZero = b.op(mlk::MathOp::Add, {x, b.integer(0, i64t)});
    MLK_CHECK(plusZero.has_value());
    b.output(*sum);
    b.output(*timesOne);
    b.output(*plusZero);
    MLK_CHECK(env.pass("property.infer")->run(env.ctx, b.graph()).has_value());
    auto r = env.pass("math.identity_elim")->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(r->changed);            // the legal int identities fired
    MLK_CHECK_EQ(b.graph().representative(*sum), *sum);      // x + 5 intact
    MLK_CHECK_EQ(b.graph().representative(*timesOne), x);    // x * 1 -> x
    MLK_CHECK_EQ(b.graph().representative(*plusZero), x);    // x + 0 -> x
}

MLK_TEST(math, cse_merges_duplicate_subtrees) {
    PassEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto sq1 = b.op(mlk::MathOp::Mul, {x, x});
    MLK_CHECK(sq1.has_value());
    auto sq2 = b.op(mlk::MathOp::Mul, {x, x});
    MLK_CHECK(sq2.has_value());
    auto sum = b.op(mlk::MathOp::Add, {*sq1, *sq2});
    MLK_CHECK(sum.has_value());
    b.output(*sum);
    auto r = env.pass("math.cse")->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(r->changed);
    MLK_CHECK_EQ(b.graph().representative(*sq2),
                 b.graph().representative(*sq1));
}

MLK_TEST(math, dce_removes_dead_pure_nodes_only) {
    PassEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto dead = b.op(mlk::MathOp::Sin, {x});  // never used
    MLK_CHECK(dead.has_value());
    auto used = b.op(mlk::MathOp::Cos, {x});
    MLK_CHECK(used.has_value());
    b.output(*used);
    auto r = env.pass("math.dce")->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(r->changed);
    MLK_CHECK(b.graph().isNodeDead(b.graph().value(*dead).producer));
    MLK_CHECK(!b.graph().isNodeDead(b.graph().value(*used).producer));
}

MLK_TEST(math, passes_are_idempotent) {
    // Rule 10: running twice produces identical IR.
    PassEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto sq = b.op(mlk::MathOp::Mul, {x, x});
    MLK_CHECK(sq.has_value());
    auto tx = b.op(mlk::MathOp::Mul, {b.constant(3.0, f64), x});
    MLK_CHECK(tx.has_value());
    auto sum = b.op(mlk::MathOp::Add, {*sq, *tx});
    MLK_CHECK(sum.has_value());
    auto s = b.op(mlk::MathOp::Sin, {*sum});
    MLK_CHECK(s.has_value());
    b.output(*s);
    auto canon = env.pass("math.canonicalize");
    MLK_CHECK(canon != nullptr);
    MLK_CHECK(canon->run(env.ctx, b.graph()).has_value());
    const mlk::HashValue h1 = mlk::graphHash(b.graph());
    const uint32_t live1 = b.graph().liveNodeCount();
    MLK_CHECK(canon->run(env.ctx, b.graph()).has_value());
    const mlk::HashValue h2 = mlk::graphHash(b.graph());
    MLK_CHECK_EQ(h1, h2);
    MLK_CHECK_EQ(live1, b.graph().liveNodeCount());
}

MLK_TEST(math, algebraic_simplify_exact_sign_identities) {
    // Every rule in algebraic_simplify must be bit-exact for ALL inputs:
    //   sub(x, +0) -> x ; div(x, 1) -> x ; div(x, -1) -> neg(x) ;
    //   neg(neg(x)) -> x. Rules outside that set stay untouched.
    PassEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto sub = b.op(mlk::MathOp::Sub, {x, b.constant(0.0, f64)});
    MLK_CHECK(sub.has_value());
    auto div1 = b.op(mlk::MathOp::Div, {x, b.constant(1.0, f64)});
    MLK_CHECK(div1.has_value());
    auto divNeg1 = b.op(mlk::MathOp::Div, {x, b.constant(-1.0, f64)});
    MLK_CHECK(divNeg1.has_value());
    auto neg1 = b.op(mlk::MathOp::Neg, {x});
    MLK_CHECK(neg1.has_value());
    auto neg2 = b.op(mlk::MathOp::Neg, {*neg1});
    MLK_CHECK(neg2.has_value());
    // x + (+0.0) is identity_elim's domain and stays UNTOUCHED here (under
    // sign preservation it must not be dropped by anyone).
    auto add = b.op(mlk::MathOp::Add, {x, b.constant(0.0, f64)});
    MLK_CHECK(add.has_value());
    b.output(*sub);
    b.output(*div1);
    b.output(*divNeg1);
    b.output(*neg2);
    b.output(*add);
    MLK_CHECK(env.pass("property.infer")->run(env.ctx, b.graph()).has_value());
    auto pass = env.pass("math.algebraic_simplify");
    MLK_CHECK(pass != nullptr);
    auto r = pass->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(r->changed);
    MLK_CHECK_EQ(b.graph().representative(*sub), x);
    MLK_CHECK_EQ(b.graph().representative(*div1), x);
    const mlk::Value& negRes =
        b.graph().value(b.graph().representative(*divNeg1));
    MLK_CHECK(negRes.kind == mlk::ValueKind::NodeResult);
    MLK_CHECK(b.graph().node(negRes.producer).op == mlk::MathOp::Neg);
    MLK_CHECK(b.graph().node(negRes.producer).inputs[0] == x);
    MLK_CHECK_EQ(b.graph().representative(*neg2), x);
    MLK_CHECK_EQ(b.graph().representative(*add), *add);  // not this pass's job
}

MLK_TEST(math, strength_reduce_power_of_two_divisor_bitexact) {
    // Div(x, 2^k) -> Mul(x, 2^-k): same real value, same rounding —
    // bit-identical for every input. Verified structurally AND by a
    // differential run against the interpreter over subnormal/zero/edge
    // samples (Rule 50: claims carry verification).
    PassEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto div8 = b.op(mlk::MathOp::Div, {x, b.constant(8.0, f64)});
    MLK_CHECK(div8.has_value());
    auto divHalf = b.op(mlk::MathOp::Div, {x, b.constant(0.5, f64)});
    MLK_CHECK(divHalf.has_value());
    auto div3 = b.op(mlk::MathOp::Div, {x, b.constant(3.0, f64)});
    MLK_CHECK(div3.has_value());
    auto div0 = b.op(mlk::MathOp::Div, {x, b.constant(0.0, f64)});
    MLK_CHECK(div0.has_value());
    b.output(*div8);
    b.output(*divHalf);
    b.output(*div3);
    b.output(*div0);
    // Types first: the pass gates on the F64 result dtype (the theorem's
    // evaluation-precision guard), so inference must run up front — same
    // as the pipeline does.
    MLK_CHECK(env.pass("type.infer")->run(env.ctx, b.graph()).has_value());
    MLK_CHECK(env.pass("property.infer")->run(env.ctx, b.graph()).has_value());
    auto pass = env.pass("math.strength_reduce");
    MLK_CHECK(pass != nullptr);
    auto r = pass->run(env.ctx, b.graph());
    MLK_CHECK(r.has_value());
    MLK_CHECK(r->changed);

    // div8 -> mul(x, 0.125)
    const mlk::ValueId repr8 = b.graph().representative(*div8);
    MLK_CHECK(repr8 != *div8);
    const mlk::Value& mulV = b.graph().value(repr8);
    MLK_CHECK(mulV.kind == mlk::ValueKind::NodeResult);
    const mlk::Node& mulN = b.graph().node(mulV.producer);
    MLK_CHECK(mulN.op == mlk::MathOp::Mul);
    MLK_CHECK_EQ(mulN.inputs[0], x);
    // The reciprocal constant (typed constant check; the internal typed
    // helpers are not part of the public test surface).
    const mlk::Value& recipV = b.graph().value(mulN.inputs[1]);
    MLK_CHECK(recipV.kind == mlk::ValueKind::Constant);
    MLK_CHECK(!recipV.constant.isInt);
    MLK_CHECK(recipV.constant.f64 == 0.125);

    // divHalf -> mul(x, 2.0)
    const mlk::Value& mulHalfV = b.graph().value(b.graph().representative(*divHalf));
    MLK_CHECK(mulHalfV.kind == mlk::ValueKind::NodeResult);
    const mlk::Node& mulHalfN = b.graph().node(mulHalfV.producer);
    MLK_CHECK(mulHalfN.op == mlk::MathOp::Mul);
    const mlk::Value& recipHalfV = b.graph().value(mulHalfN.inputs[1]);
    MLK_CHECK(recipHalfV.kind == mlk::ValueKind::Constant);
    MLK_CHECK(!recipHalfV.constant.isInt);
    MLK_CHECK(recipHalfV.constant.f64 == 2.0);

    // Non-power-of-two and zero divisors stay divisions.
    const mlk::Value& div3V = b.graph().value(b.graph().representative(*div3));
    MLK_CHECK(div3V.kind == mlk::ValueKind::NodeResult);
    MLK_CHECK(b.graph().node(div3V.producer).op == mlk::MathOp::Div);
    const mlk::Value& div0V = b.graph().value(b.graph().representative(*div0));
    MLK_CHECK(b.graph().node(div0V.producer).op == mlk::MathOp::Div);

    // Differential bit-exactness: the rewritten graph must produce the
    // same bits as the original form for every sample.
    const std::size_t kSamples = 8;
    const double samples[kSamples] = {
        5e-324,                      // min subnormal
        1e-320,                      // subnormal
        2.2250738585072014e-308,     // DBL_MIN
        -0.0,
        0.0,
        1.0,
        1.7976931348623157e308,      // DBL_MAX (exact /8, *0.125)
        -2.5e-300,
    };
    mlk::SmallVector<double, 8> inputs;
    for (std::size_t i = 0; i < kSamples; ++i) inputs.push_back(samples[i]);
    // Original-form graph (fresh copy of the same four divisions).
    mlk::GraphBuilder orig(env.symbols);
    const mlk::ValueId ox = orig.placeholder("x", f64);
    auto o8 = orig.op(mlk::MathOp::Div, {ox, orig.constant(8.0, f64)});
    MLK_CHECK(o8.has_value());
    auto oh = orig.op(mlk::MathOp::Div, {ox, orig.constant(0.5, f64)});
    MLK_CHECK(oh.has_value());
    orig.output(*o8);
    orig.output(*oh);
    auto before = mlk::interpretGraph(orig.graph(), inputs);
    MLK_CHECK(before.has_value());
    // Rebuild a rewritten-form graph by re-running the pass on a fresh
    // identical graph, then compare outputs bit-exactly.
    mlk::GraphBuilder b2(env.symbols);
    const mlk::ValueId x2 = b2.placeholder("x", f64);
    auto d82 = b2.op(mlk::MathOp::Div, {x2, b2.constant(8.0, f64)});
    MLK_CHECK(d82.has_value());
    auto dh2 = b2.op(mlk::MathOp::Div, {x2, b2.constant(0.5, f64)});
    MLK_CHECK(dh2.has_value());
    b2.output(*d82);
    b2.output(*dh2);
    MLK_CHECK(env.pass("property.infer")->run(env.ctx, b2.graph()).has_value());
    MLK_CHECK(env.pass("math.strength_reduce")->run(env.ctx, b2.graph()).has_value());
    auto after = mlk::interpretGraph(b2.graph(), inputs);
    MLK_CHECK(after.has_value());
    for (std::size_t i = 0; i < before->outputScalars.size(); ++i) {
        const double a = before->outputScalars[i];
        const double bval = after->outputScalars[i];
        // Bit-exact: identical value AND identical sign (distinguishes -0).
        MLK_CHECK(a == bval);
        MLK_CHECK(std::signbit(a) == std::signbit(bval));
    }
}

MLK_TEST(math, canonicalize_int_graph_not_miscompiled_end_to_end) {
    // Regression for the certified int miscompile: add(x_i64, 5) used to
    // fold to x through the untyped constant probe. The full canonicalize
    // composition must leave both the graph AND the interpreted value
    // intact (interpreter evaluates in f64; 1 + 5 == 6 either way).
    PassEnv env;
    mlk::GraphBuilder b(env.symbols);
    const mlk::MathType i64t =
        mlk::MathType::scalar(mlk::Domain::Int, mlk::Dtype::I64);
    const mlk::ValueId x = b.placeholder("x", i64t);
    auto sum = b.op(mlk::MathOp::Add, {x, b.integer(5, i64t)});
    MLK_CHECK(sum.has_value());
    b.output(*sum);
    MLK_CHECK(env.pass("property.infer")->run(env.ctx, b.graph()).has_value());
    auto canon = env.pass("math.canonicalize");
    MLK_CHECK(canon != nullptr);
    MLK_CHECK(canon->run(env.ctx, b.graph()).has_value());
    MLK_CHECK_EQ(b.graph().representative(*sum), *sum);  // intact
    mlk::SmallVector<double, 8> inputs{1.0};
    auto out = mlk::interpretGraph(b.graph(), inputs);
    MLK_CHECK(out.has_value());
    MLK_CHECK_EQ(out->outputScalars[0], 6.0);
}

MLK_TEST(math, canonicalize_neg_zero_semantics_preserved_end_to_end) {
    // Regression for the certified -0 flip: canonicalize must not change
    // add(x, +0.0) under sign preservation, so execution at x = -0 keeps
    // the IEEE result +0.0. The always-sound add(x, -0.0) folds and stays
    // bit-equal (-0 + -0 = -0).
    PassEnv env;  // preserveNegativeZero = true (default)
    {
        mlk::GraphBuilder b(env.symbols);
        const mlk::MathType f64 =
            mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
        const mlk::ValueId x = b.placeholder("x", f64);
        auto sum = b.op(mlk::MathOp::Add, {x, b.constant(0.0, f64)});
        MLK_CHECK(sum.has_value());
        b.output(*sum);
        MLK_CHECK(
            env.pass("property.infer")->run(env.ctx, b.graph()).has_value());
        MLK_CHECK(env.pass("math.canonicalize")->run(env.ctx, b.graph()).has_value());
        MLK_CHECK_EQ(b.graph().representative(*sum), *sum);  // NOT rewritten
        mlk::SmallVector<double, 8> inputs{-0.0};
        auto out = mlk::interpretGraph(b.graph(), inputs);
        MLK_CHECK(out.has_value());
        MLK_CHECK(out->outputScalars[0] == 0.0);
        MLK_CHECK(!std::signbit(out->outputScalars[0]));  // +0, not -0
    }
    {
        mlk::GraphBuilder b(env.symbols);
        const mlk::MathType f64 =
            mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
        const mlk::ValueId x = b.placeholder("x", f64);
        auto sum = b.op(mlk::MathOp::Add, {x, b.constant(-0.0, f64)});
        MLK_CHECK(sum.has_value());
        b.output(*sum);
        MLK_CHECK(
            env.pass("property.infer")->run(env.ctx, b.graph()).has_value());
        MLK_CHECK(env.pass("math.canonicalize")->run(env.ctx, b.graph()).has_value());
        MLK_CHECK_EQ(b.graph().representative(*sum), x);  // folded: exact
        mlk::SmallVector<double, 8> inputs{-0.0};
        auto out = mlk::interpretGraph(b.graph(), inputs);
        MLK_CHECK(out.has_value());
        MLK_CHECK(out->outputScalars[0] == 0.0);
        MLK_CHECK(std::signbit(out->outputScalars[0]));  // -0 preserved
    }
}

MLK_TEST_MAIN("math")
