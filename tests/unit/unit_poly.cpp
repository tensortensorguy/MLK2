// Polyhedral engine unit tests (Rules 10, 33, 42, 90): exact rational
// arithmetic, affine expressions, Presburger set operations, FM emptiness,
// lexmin/lexmax witness verification, affine map image/preimage.
#include <optional>

#include "mlk/poly/poly.h"
#include "mlk/runtime/execution.h"
#include "mlk/backend/cpp_emitter.h"
#include "mlk/backend/asm_emitter.h"
#include "mlk/backend/backend_driver.h"

#include <cmath>
#include <cstdlib>

#include "mlk_test.h"

namespace {

using mlk::SmallVector;
using mlk::poly::AffineExpr;
using mlk::poly::AffineMap;
using mlk::poly::Polyhedron;
using mlk::poly::PresburgerSet;
using mlk::poly::Rational;
using mlk::poly::VarSpace;

/// Test helper: normalized rational with a hard failure on bad parts.
Rational rat(int64_t n, int64_t d) {
    auto r = Rational::fromParts(n, d);
    MLK_CHECK(r.has_value());
    return r.has_value() ? *r : Rational{};
}

/// 2-dim, no-symbol space shared by most tests.
VarSpace dims2() { return VarSpace{2, 0}; }

SmallVector<int64_t, 8> pt2(int64_t a, int64_t b) {
    SmallVector<int64_t, 8> v;
    v.push_back(a);
    v.push_back(b);
    return v;
}

SmallVector<int64_t, 8> pt3(int64_t a, int64_t b, int64_t c) {
    SmallVector<int64_t, 8> v;
    v.push_back(a);
    v.push_back(b);
    v.push_back(c);
    return v;
}

MLK_TEST(poly, rational_exact_arithmetic) {
    const Rational a = rat(1, 3);
    const Rational b = rat(1, 6);
    auto sum = mlk::poly::ratAdd(a, b);
    MLK_CHECK(sum.has_value());
    MLK_CHECK_EQ(sum->num, 1);
    MLK_CHECK_EQ(sum->den, 2);

    auto prod = mlk::poly::ratMul(rat(2, 3), rat(3, 4));
    MLK_CHECK(prod.has_value());
    MLK_CHECK_EQ(prod->num, 1);
    MLK_CHECK_EQ(prod->den, 2);

    auto neg = mlk::poly::ratNeg(rat(3, 5));
    MLK_CHECK(neg.has_value());
    MLK_CHECK_EQ(neg->num, -3);

    // Division by zero is an error, not a crash (Rule 6).
    auto div0 = mlk::poly::ratDiv(rat(1, 2), rat(0, 7));
    MLK_CHECK(!div0.has_value());

    // Ordering is total and exact.
    MLK_CHECK(mlk::poly::ratLess(rat(1, 3), rat(2, 3)));
    MLK_CHECK(!mlk::poly::ratEqual(rat(1, 2), rat(1, 3)));
    MLK_CHECK(mlk::poly::ratEqual(rat(2, 4), rat(1, 2)));

    // Zero normalizes to 0/1.
    const Rational z = rat(0, 5);
    MLK_CHECK(z.isZero());
    MLK_CHECK_EQ(z.den, 1);
}

MLK_TEST(poly, rational_floor_ceil) {
    auto f = mlk::poly::ratFloor(rat(-7, 2));
    MLK_CHECK(f.has_value());
    MLK_CHECK_EQ(*f, -4);  // floor(-3.5) = -4
    auto c = mlk::poly::ratCeil(rat(-7, 2));
    MLK_CHECK(c.has_value());
    MLK_CHECK_EQ(*c, -3);  // ceil(-3.5) = -3
    auto f2 = mlk::poly::ratFloor(rat(7, 2));
    MLK_CHECK(f2.has_value());
    MLK_CHECK_EQ(*f2, 3);
    auto c2 = mlk::poly::ratCeil(rat(7, 2));
    MLK_CHECK(c2.has_value());
    MLK_CHECK_EQ(*c2, 4);
    // Exact integers floor/ceil to themselves.
    auto f3 = mlk::poly::ratFloor(rat(-6, 2));
    MLK_CHECK(f3.has_value());
    MLK_CHECK_EQ(*f3, -3);
}

MLK_TEST(poly, affine_expr_eval_and_limits) {
    const VarSpace sp = dims2();
    // e = 3*i + 2*j + 1
    auto t1 = mlk::poly::exprScale(AffineExpr::variable(sp, 0), 3);
    MLK_CHECK(t1.has_value());
    auto t2 = mlk::poly::exprScale(AffineExpr::variable(sp, 1), 2);
    MLK_CHECK(t2.has_value());
    auto s1 = mlk::poly::exprAdd(*t1, *t2);
    MLK_CHECK(s1.has_value());
    auto s2 = mlk::poly::exprAdd(*s1, AffineExpr::fromConstant(sp, 1));
    MLK_CHECK(s2.has_value());
    const AffineExpr e = *s2;
    SmallVector<int64_t, 8> vars;
    vars.push_back(2);
    vars.push_back(5);
    auto v = e.evalAt(vars);
    MLK_CHECK(v.has_value());
    MLK_CHECK_EQ(*v, 3 * 2 + 2 * 5 + 1);
    int64_t cst = 0;
    MLK_CHECK(!e.isConstant(&cst));
    MLK_CHECK(AffineExpr::fromConstant(sp, 7).isConstant(&cst));
    MLK_CHECK_EQ(cst, 7);
    // Space mismatch is rejected (Rule 67).
    auto bad = mlk::poly::exprAdd(e, AffineExpr::variable(VarSpace{3, 0}, 0));
    MLK_CHECK(!bad.has_value());
    // Structural equality.
    auto again = mlk::poly::exprAdd(*s1, AffineExpr::fromConstant(sp, 1));
    MLK_CHECK(again.has_value());
    MLK_CHECK(mlk::poly::exprEqual(e, *again));
}

MLK_TEST(poly, box_basics_and_contains) {
    const VarSpace sp = dims2();
    auto box = PresburgerSet::box(sp, pt2(0, 0), pt2(3, 5));
    MLK_CHECK(box.has_value());
    MLK_CHECK(box->containsPoint(pt2(0, 0)));
    MLK_CHECK(box->containsPoint(pt2(3, 5)));
    MLK_CHECK(box->containsPoint(pt2(2, 4)));
    MLK_CHECK(!box->containsPoint(pt2(4, 0)));
    MLK_CHECK(!box->containsPoint(pt2(-1, 2)));
    MLK_CHECK(!box->containsPoint(pt2(1, 6)));
}

MLK_TEST(poly, intersect_union_subtract) {
    const VarSpace sp = dims2();
    auto a = PresburgerSet::box(sp, pt2(0, 0), pt2(2, 2));
    auto b = PresburgerSet::box(sp, pt2(1, 1), pt2(4, 4));
    MLK_CHECK(a.has_value() && b.has_value());

    auto inter = PresburgerSet::intersect(*a, *b);
    MLK_CHECK(inter.has_value());
    MLK_CHECK(inter->containsPoint(pt2(1, 1)));
    MLK_CHECK(inter->containsPoint(pt2(2, 2)));
    MLK_CHECK(!inter->containsPoint(pt2(0, 0)));
    MLK_CHECK(!inter->containsPoint(pt2(3, 3)));

    auto uni = PresburgerSet::unite(*a, *b);
    MLK_CHECK(uni.has_value());
    MLK_CHECK(uni->containsPoint(pt2(0, 0)));
    MLK_CHECK(uni->containsPoint(pt2(4, 4)));
    MLK_CHECK(uni->containsPoint(pt2(3, 1)));

    auto diff = PresburgerSet::subtract(*b, *a);
    MLK_CHECK(diff.has_value());
    MLK_CHECK(diff->containsPoint(pt2(3, 3)));
    MLK_CHECK(diff->containsPoint(pt2(4, 4)));
    MLK_CHECK(!diff->containsPoint(pt2(2, 2)));
    MLK_CHECK(!diff->containsPoint(pt2(0, 0)));

    // A ∩ ~B via complement agrees with subtract on the corner.
    auto notb = PresburgerSet::complement(*b);
    MLK_CHECK(notb.has_value());
    auto aNotB = PresburgerSet::intersect(*a, *notb);
    MLK_CHECK(aNotB.has_value());
    MLK_CHECK(aNotB->containsPoint(pt2(0, 0)));
    MLK_CHECK(!aNotB->containsPoint(pt2(1, 1)));
}

MLK_TEST(poly, fm_emptiness_basic) {
    const VarSpace sp = dims2();
    // Empty box: [2,1] x [0,0] (lower > upper).
    auto empty = PresburgerSet::box(sp, pt2(2, 0), pt2(1, 0));
    MLK_CHECK(empty.has_value());
    auto f = empty->provablyEmpty();
    MLK_CHECK(f.has_value());
    MLK_CHECK(*f);

    // Non-empty box is provably non-empty.
    auto full = PresburgerSet::box(sp, pt2(0, 0), pt2(1, 0));
    MLK_CHECK(full.has_value());
    auto nf = full->provablyEmpty();
    MLK_CHECK(nf.has_value());
    MLK_CHECK(!*nf);
}

MLK_TEST(poly, fm_emptiness_diagonal_cross) {
    // x - y >= 2 and y - x >= 2 are contradictory.
    const VarSpace sp = dims2();
    Polyhedron p;
    p.space = sp;
    SmallVector<int64_t, 8> r1;
    r1.push_back(1);
    r1.push_back(-1);
    p.addInequality(std::move(r1), -2);
    SmallVector<int64_t, 8> r2;
    r2.push_back(-1);
    r2.push_back(1);
    p.addInequality(std::move(r2), -2);
    PresburgerSet s;
    s.space = sp;
    s.disjuncts.push_back(std::move(p));
    auto f = s.provablyEmpty();
    MLK_CHECK(f.has_value());
    MLK_CHECK(*f);
}

MLK_TEST(poly, fm_emptiness_stripped_triangle) {
    // x >= 1, y >= 1, x + y <= 2 → single integer point (1,1); not empty.
    // Shrink the sum bound to 1 → empty.
    const VarSpace sp = dims2();
    auto build = [sp](int64_t sumBound) {
        Polyhedron p;
        p.space = sp;
        SmallVector<int64_t, 8> rx;
        rx.push_back(1);
        rx.push_back(0);
        p.addInequality(std::move(rx), -1);  // x - 1 >= 0
        SmallVector<int64_t, 8> ry;
        ry.push_back(0);
        ry.push_back(1);
        p.addInequality(std::move(ry), -1);  // y - 1 >= 0
        SmallVector<int64_t, 8> rs;
        rs.push_back(-1);
        rs.push_back(-1);
        p.addInequality(std::move(rs), sumBound);  // B - x - y >= 0
        PresburgerSet s;
        s.space = sp;
        s.disjuncts.push_back(std::move(p));
        return s;
    };
    auto f1 = build(2).provablyEmpty();
    MLK_CHECK(f1.has_value());
    MLK_CHECK(!*f1);
    auto f2 = build(1).provablyEmpty();
    MLK_CHECK(f2.has_value());
    MLK_CHECK(*f2);
}

MLK_TEST(poly, exists_projection_diagonal_band) {
    // {(x,y,t) : x <= t <= x+3, y == t} → projected: x <= y <= x+3.
    const VarSpace sp3 = VarSpace{3, 0};
    Polyhedron p;
    p.space = sp3;
    SmallVector<int64_t, 8> r1;
    r1.push_back(-1);
    r1.push_back(0);
    r1.push_back(1);
    p.addInequality(std::move(r1), 0);  // t - x >= 0
    SmallVector<int64_t, 8> r2;
    r2.push_back(1);
    r2.push_back(0);
    r2.push_back(-1);
    p.addInequality(std::move(r2), 3);  // x + 3 - t >= 0
    SmallVector<int64_t, 8> r3;
    r3.push_back(0);
    r3.push_back(1);
    r3.push_back(-1);
    p.addEquality(std::move(r3), 0);  // y - t == 0
    PresburgerSet s;
    s.space = sp3;
    s.disjuncts.push_back(std::move(p));
    auto proj = s.projectOut(2);
    MLK_CHECK(proj.has_value());
    MLK_CHECK(!proj->disjuncts.empty());
    if (!proj->disjuncts.empty()) {
        MLK_CHECK_EQ(proj->disjuncts[0].space.nDims, 3);  // columns kept
        // Eliminated column is zero: t value is irrelevant at contains.
        MLK_CHECK(proj->containsPoint(pt3(0, 3, 0)));
        MLK_CHECK(proj->containsPoint(pt3(2, 5, 0)));
        MLK_CHECK(!proj->containsPoint(pt3(0, 4, 0)));
        MLK_CHECK(!proj->containsPoint(pt3(1, 0, 0)));
    }
}

MLK_TEST(poly, lexmin_lexmax_box) {
    const VarSpace sp = dims2();
    auto box = PresburgerSet::box(sp, pt2(-2, 3), pt2(5, 9));
    MLK_CHECK(box.has_value());
    auto mn = box->lexMin();
    MLK_CHECK(mn.has_value());
    MLK_CHECK(mn->has_value());
    if (mn->has_value()) {
        MLK_CHECK_EQ((**mn)[0], -2);
        MLK_CHECK_EQ((**mn)[1], 3);
    }
    auto mx = box->lexMax();
    MLK_CHECK(mx.has_value());
    MLK_CHECK(mx->has_value());
    if (mx->has_value()) {
        MLK_CHECK_EQ((**mx)[0], 5);
        MLK_CHECK_EQ((**mx)[1], 9);
    }
}

MLK_TEST(poly, lexmin_lexmax_coupled_bounds) {
    // 0 <= x,y <= 7 with y >= x and x + y <= 9:
    // lexmin = (0, 0); lexmax = (4, 5).
    const VarSpace sp = dims2();
    Polyhedron p;
    p.space = sp;
    SmallVector<int64_t, 8> rx0;
    rx0.push_back(1);
    rx0.push_back(0);
    p.addInequality(std::move(rx0), 0);
    SmallVector<int64_t, 8> ry0;
    ry0.push_back(0);
    ry0.push_back(1);
    p.addInequality(std::move(ry0), 0);
    SmallVector<int64_t, 8> rx7;
    rx7.push_back(-1);
    rx7.push_back(0);
    p.addInequality(std::move(rx7), 7);
    SmallVector<int64_t, 8> ry7;
    ry7.push_back(0);
    ry7.push_back(-1);
    p.addInequality(std::move(ry7), 7);
    SmallVector<int64_t, 8> rge;
    rge.push_back(-1);
    rge.push_back(1);
    p.addInequality(std::move(rge), 0);
    SmallVector<int64_t, 8> rsum;
    rsum.push_back(-1);
    rsum.push_back(-1);
    p.addInequality(std::move(rsum), 9);
    PresburgerSet s;
    s.space = sp;
    s.disjuncts.push_back(std::move(p));

    auto mn = s.lexMin();
    MLK_CHECK(mn.has_value());
    MLK_CHECK(mn->has_value());
    if (mn->has_value()) {
        MLK_CHECK_EQ((**mn)[0], 0);
        MLK_CHECK_EQ((**mn)[1], 0);
    }
    auto mx = s.lexMax();
    MLK_CHECK(mx.has_value());
    MLK_CHECK(mx->has_value());
    if (mx->has_value()) {
        MLK_CHECK_EQ((**mx)[0], 4);
        MLK_CHECK_EQ((**mx)[1], 5);
        MLK_CHECK(s.containsPoint(**mx));  // witness verification
    }
}

MLK_TEST(poly, lexmin_empty_set) {
    const VarSpace sp = dims2();
    auto empty = PresburgerSet::box(sp, pt2(5, 5), pt2(1, 9));
    MLK_CHECK(empty.has_value());
    auto mn = empty->lexMin();
    MLK_CHECK(mn.has_value());
    MLK_CHECK(!mn->has_value());  // nullopt = integer-empty
}

MLK_TEST(poly, lexmin_requires_specialized_symbols) {
    const VarSpace sp = VarSpace{1, 1};  // one dim + one symbol
    auto box = PresburgerSet::box(sp, SmallVector<int64_t, 8>{0},
                                  SmallVector<int64_t, 8>{4});
    MLK_CHECK(box.has_value());
    // Contract: lexMin errors when symbols are not specialized.
    auto mn = box->lexMin();
    MLK_CHECK(!mn.has_value());
    auto mx = box->lexMax();
    MLK_CHECK(!mx.has_value());
}

MLK_TEST(poly, map_image_permutation) {
    // Domain: 0 <= i,j <= 3; map f(i,j) = (j, i). Image = same box.
    const VarSpace sp = dims2();
    auto dom = PresburgerSet::box(sp, pt2(0, 0), pt2(3, 3));
    MLK_CHECK(dom.has_value());
    AffineMap swap;
    swap.inSpace = sp;
    swap.nOut = 2;
    swap.outputs.push_back(AffineExpr::variable(sp, 1));
    swap.outputs.push_back(AffineExpr::variable(sp, 0));
    auto img = mlk::poly::mapImage(*dom, swap);
    MLK_CHECK(img.has_value());
    if (img.has_value()) {
        MLK_CHECK_EQ(img->space.nDims, 2);
        MLK_CHECK_EQ(img->space.nSyms, 0);
        MLK_CHECK(img->containsPoint(pt2(0, 0)));
        MLK_CHECK(img->containsPoint(pt2(3, 3)));
        MLK_CHECK(!img->containsPoint(pt2(4, 0)));
        auto mn = img->lexMin();
        MLK_CHECK(mn.has_value() && mn->has_value());
        if (mn.has_value() && mn->has_value()) {
            MLK_CHECK_EQ((**mn)[0], 0);
            MLK_CHECK_EQ((**mn)[1], 0);
        }
    }
}

MLK_TEST(poly, map_image_shrinking) {
    // Domain: 0 <= i,j <= 3; map f(i,j) = (i + j). Image: 0 <= t <= 6.
    const VarSpace sp = dims2();
    auto dom = PresburgerSet::box(sp, pt2(0, 0), pt2(3, 3));
    MLK_CHECK(dom.has_value());
    AffineMap sum;
    sum.inSpace = sp;
    sum.nOut = 1;
    auto t1 = mlk::poly::exprAdd(AffineExpr::variable(sp, 0),
                                 AffineExpr::variable(sp, 1));
    MLK_CHECK(t1.has_value());
    sum.outputs.push_back(*t1);
    auto img = mlk::poly::mapImage(*dom, sum);
    MLK_CHECK(img.has_value());
    if (img.has_value()) {
        MLK_CHECK_EQ(img->space.nDims, 1);
        SmallVector<int64_t, 8> pt0;
        pt0.push_back(0);
        SmallVector<int64_t, 8> pt6;
        pt6.push_back(6);
        SmallVector<int64_t, 8> pt7;
        pt7.push_back(7);
        MLK_CHECK(img->containsPoint(pt0));
        MLK_CHECK(img->containsPoint(pt6));
        MLK_CHECK(!img->containsPoint(pt7));
    }
}

MLK_TEST(poly, map_preimage_guard_region) {
    // f(i) = (i, 2i); range: 0 <= t0 <= 3, 0 <= t1 <= 8.
    // Preimage: 0 <= i <= 3 (t0 binds before t1).
    const VarSpace sp = VarSpace{1, 0};
    const VarSpace sp2 = dims2();
    Polyhedron rp;
    rp.space = sp2;
    SmallVector<int64_t, 8> c0;
    c0.push_back(-1);
    c0.push_back(0);
    rp.addInequality(std::move(c0), 3);  // -t0 + 3 >= 0
    SmallVector<int64_t, 8> c1;
    c1.push_back(0);
    c1.push_back(-1);
    rp.addInequality(std::move(c1), 8);  // -t1 + 8 >= 0
    SmallVector<int64_t, 8> d0;
    d0.push_back(1);
    d0.push_back(0);
    rp.addInequality(std::move(d0), 0);
    SmallVector<int64_t, 8> d1;
    d1.push_back(0);
    d1.push_back(1);
    rp.addInequality(std::move(d1), 0);
    PresburgerSet range;
    range.space = sp2;
    range.disjuncts.push_back(std::move(rp));

    AffineMap f;
    f.inSpace = sp;
    f.nOut = 2;
    f.outputs.push_back(AffineExpr::variable(sp, 0));
    auto sc = mlk::poly::exprScale(AffineExpr::variable(sp, 0), 2);
    MLK_CHECK(sc.has_value());
    f.outputs.push_back(*sc);

    auto pre = mlk::poly::mapPreimage(range, f);
    MLK_CHECK(pre.has_value());
    if (pre.has_value()) {
        MLK_CHECK_EQ(pre->space.nDims, 1);
        SmallVector<int64_t, 8> i0;
        i0.push_back(0);
        SmallVector<int64_t, 8> i3;
        i3.push_back(3);
        SmallVector<int64_t, 8> i4;
        i4.push_back(4);
        MLK_CHECK(pre->containsPoint(i0));
        MLK_CHECK(pre->containsPoint(i3));
        MLK_CHECK(!pre->containsPoint(i4));  // t0 = 4 > 3
    }
}

MLK_TEST(poly, map_compose_affine) {
    // inner: (i,j) -> (i, j); outer: (a,b) -> (a+b, b). Compose = (i+j, j).
    const VarSpace sp = dims2();
    AffineMap inner = AffineMap::identity(sp);
    AffineMap outer;
    outer.inSpace = sp;
    outer.nOut = 2;
    auto o0 = mlk::poly::exprAdd(AffineExpr::variable(sp, 0),
                                 AffineExpr::variable(sp, 1));
    MLK_CHECK(o0.has_value());
    outer.outputs.push_back(*o0);
    outer.outputs.push_back(AffineExpr::variable(sp, 1));
    auto comp = mlk::poly::mapCompose(outer, inner);
    MLK_CHECK(comp.has_value());
    if (comp.has_value()) {
        SmallVector<int64_t, 8> in;
        in.push_back(2);
        in.push_back(3);
        auto v = comp->evalAt(in);
        MLK_CHECK(v.has_value());
        if (v.has_value()) {
            MLK_CHECK_EQ((*v)[0], 5);
            MLK_CHECK_EQ((*v)[1], 3);
        }
    }
}

MLK_TEST(poly, expression_gcd_normalize) {
    const VarSpace sp = dims2();
    auto a = mlk::poly::exprScale(AffineExpr::variable(sp, 0), 4);
    MLK_CHECK(a.has_value());
    auto b = mlk::poly::exprScale(AffineExpr::variable(sp, 1), 6);
    MLK_CHECK(b.has_value());
    auto ab = mlk::poly::exprAdd(*a, *b);
    MLK_CHECK(ab.has_value());
    auto abc = mlk::poly::exprAdd(*ab, AffineExpr::fromConstant(sp, 8));
    MLK_CHECK(abc.has_value());
    AffineExpr e = *abc;
    e.gcdNormalize();
    MLK_CHECK_EQ(e.coeffs[0], 2);
    MLK_CHECK_EQ(e.coeffs[1], 3);
    MLK_CHECK_EQ(e.constant, 4);
}

MLK_TEST(poly, set_with_symbols_emptiness_parametric) {
    // Parametric: 0 <= i < N (symbol N), i >= 5 → empty iff N <= 5.
    // The engine answers over ALL symbol values (existential), so the set
    // is NonEmpty (N = 6 works). Dependence legality relies on this.
    const VarSpace sp = VarSpace{1, 1};
    PresburgerSet s;
    s.space = sp;
    Polyhedron p;
    p.space = sp;
    SmallVector<int64_t, 8> ri;
    ri.push_back(1);
    ri.push_back(0);
    p.addInequality(std::move(ri), 0);  // i >= 0
    SmallVector<int64_t, 8> rn;
    rn.push_back(-1);
    rn.push_back(1);  // N - i - 1 >= 0  (i < N)
    p.addInequality(std::move(rn), -1);
    SmallVector<int64_t, 8> r5;
    r5.push_back(1);
    r5.push_back(0);
    p.addInequality(std::move(r5), -5);  // i - 5 >= 0
    s.disjuncts.push_back(std::move(p));
    auto f = s.provablyEmpty();
    MLK_CHECK(f.has_value());
    MLK_CHECK(f.has_value() && !*f);  // not provably empty => NonEmpty
    // Shrink the band: i >= 0 and i < N and i <= -1 → i >= 0 and i < N and
    // N >= 1... instead force contradiction: i >= 1 and i <= 0 (pure dims).
    const VarSpace sp1 = VarSpace{1, 0};
    Polyhedron q;
    q.space = sp1;
    SmallVector<int64_t, 8> q1;
    q1.push_back(1);
    q.addInequality(std::move(q1), -1);  // i - 1 >= 0  → i >= 1
    SmallVector<int64_t, 8> q2;
    q2.push_back(-1);
    q.addInequality(std::move(q2), 0);  // -i >= 0  → i <= 0 (contradiction)
    PresburgerSet qs;
    qs.space = sp1;
    qs.disjuncts.push_back(std::move(q));
    auto fq = qs.provablyEmpty();
    MLK_CHECK(fq.has_value());
    MLK_CHECK(*fq);
}

}  // namespace

// --- SCoP extraction (iteration 2) -------------------------------------------

#include "mlk/core/diagnostics.h"
#include "mlk/pass/pass_registry.h"
#include "mlk/pass/register_all.h"
#include "mlk/pipeline/pipeline_runner.h"

using mlk::DiagnosticEngine;
using mlk::KernelBuffer;
using mlk::KernelExpr;
using mlk::KernelModule;
using mlk::KernelNode;
using mlk::KernelOperand;
using mlk::SymbolId;
using mlk::SymbolTable;
using mlk::poly::extractScop;
using mlk::poly::Scop;
using mlk::poly::Statement;

/// Test constants (Rule 27): gemm shapes M=4, K=3, N=5.
constexpr int64_t kTestM = 4;
constexpr int64_t kTestK = 3;
constexpr int64_t kTestN = 5;

/// Builds the init+accumulate GEMM nest:
///   S0 (depth 2): C[i][j] = 0                       over i,j
///   S1 (depth 3): C[i][j] += A[i][k] * B[k][j]      over i,j,k
[[nodiscard]] KernelModule buildGemmKernel(SymbolTable& symbols) {
    KernelModule km;
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = SmallVector<int64_t, 4>{kTestM, kTestK};
    a.isInput = true;
    KernelBuffer b;
    b.name = symbols.intern("B");
    b.dims = SmallVector<int64_t, 4>{kTestK, kTestN};
    b.isInput = true;
    KernelBuffer c;
    c.name = symbols.intern("C");
    c.dims = SmallVector<int64_t, 4>{kTestM, kTestN};
    c.isOutput = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufB = km.addBuffer(b);
    const uint32_t bufC = km.addBuffer(c);
    const SymbolId vi = symbols.intern("i");
    const SymbolId vj = symbols.intern("j");
    const SymbolId vk = symbols.intern("k");

    // S0: compute chain [Const 0], store C[i][j] (flat i*N + j).
    KernelNode compute0;
    compute0.op = mlk::KernelOp::Compute;
    KernelExpr zero;
    zero.op = mlk::MathOp::Add;  // unary ignored; operand is the value
    zero.a.kind = KernelOperand::Kind::Const;
    zero.a.constValue = 0.0;
    compute0.exprs.push_back(zero);
    KernelNode store0;
    store0.op = mlk::KernelOp::Store;
    store0.bufferOut = bufC;
    store0.outIndexCoeffs = SmallVector<int64_t, 4>{kTestN, 1};
    store0.outIndexOffset = 0;

    // S1: compute chain [t0 = A[i,k], t1 = B[k,j], t2 = t0*t1], store
    // C[i][j] with accumulate = true.
    KernelNode compute1;
    compute1.op = mlk::KernelOp::Compute;
    KernelExpr ea;
    ea.op = mlk::MathOp::Add;
    ea.a.kind = KernelOperand::Kind::ElemIdx;
    ea.a.index = static_cast<int64_t>(bufA);
    ea.a.idxCoeffs = SmallVector<int64_t, 4>{kTestK, 0, 1};  // i*K + k
    KernelExpr eb;
    eb.op = mlk::MathOp::Add;
    eb.a.kind = KernelOperand::Kind::ElemIdx;
    eb.a.index = static_cast<int64_t>(bufB);
    eb.a.idxCoeffs = SmallVector<int64_t, 4>{0, 1, kTestN};  // k*N + j
    KernelExpr mul;
    mul.op = mlk::MathOp::Mul;
    mul.a.kind = KernelOperand::Kind::Temp;
    mul.a.index = 0;
    mul.b.kind = KernelOperand::Kind::Temp;
    mul.b.index = 1;
    compute1.exprs.push_back(ea);
    compute1.exprs.push_back(eb);
    compute1.exprs.push_back(mul);
    KernelNode store1;
    store1.op = mlk::KernelOp::Store;
    store1.bufferOut = bufC;
    store1.outIndexCoeffs = SmallVector<int64_t, 4>{kTestN, 1, 0};
    store1.accumulate = true;

    // Loop tree: i { j { [S0 pair] , k { [S1 pair] } } }.
    KernelNode kLoop;
    kLoop.op = mlk::KernelOp::Loop;
    kLoop.var = vk;
    kLoop.begin = 0;
    kLoop.end = kTestK;
    {
        uint32_t c1 = km.addNode(compute1);
        uint32_t s1 = km.addNode(store1);
        kLoop.children.push_back(c1);
        kLoop.children.push_back(s1);
    }
    KernelNode jLoop;
    jLoop.op = mlk::KernelOp::Loop;
    jLoop.var = vj;
    jLoop.begin = 0;
    jLoop.end = kTestN;
    {
        uint32_t c0 = km.addNode(compute0);
        uint32_t s0 = km.addNode(store0);
        uint32_t kId = km.addNode(kLoop);
        jLoop.children.push_back(c0);
        jLoop.children.push_back(s0);
        jLoop.children.push_back(kId);
    }
    KernelNode iLoop;
    iLoop.op = mlk::KernelOp::Loop;
    iLoop.var = vi;
    iLoop.begin = 0;
    iLoop.end = kTestM;
    {
        uint32_t jId = km.addNode(jLoop);
        iLoop.children.push_back(jId);
    }
    (void)km.addNode(iLoop);
    return km;
}

MLK_TEST(poly, scop_extract_gemm) {
    SymbolTable symbols;
    KernelModule km = buildGemmKernel(symbols);
    auto scop = extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;
    MLK_CHECK_EQ(scop->depth, 3);
    MLK_CHECK_EQ(scop->space.nDims, 3);
    MLK_CHECK_EQ(scop->space.nSyms, 0);
    MLK_CHECK_EQ(scop->statements.size(), 2);

    const Statement& s0 = scop->statements[0];
    const Statement& s1 = scop->statements[1];
    MLK_CHECK_EQ(s0.depth, 2);  // init: i,j only
    MLK_CHECK_EQ(s1.depth, 3);  // accumulate: i,j,k
    MLK_CHECK_EQ(s0.accesses.size(), 1);
    MLK_CHECK_EQ(s1.accesses.size(), 4);  // A, B, write C, implicit read C
    MLK_CHECK(s0.accesses[0].isWrite);
    MLK_CHECK(!s1.accesses[0].isWrite);
    MLK_CHECK(!s1.accesses[1].isWrite);
    MLK_CHECK(s1.accesses[2].isWrite);
    MLK_CHECK(!s1.accesses[3].isWrite);  // accumulate's implicit read
    MLK_CHECK(s1.accumulate);

    // Domain spot checks (full rank; deeper dims pinned to 0 for S0).
    SmallVector<int64_t, 8> p0;
    p0.push_back(1);
    p0.push_back(2);
    p0.push_back(0);
    MLK_CHECK(s0.domain.containsPoint(p0));
    SmallVector<int64_t, 8> p0bad;
    p0bad.push_back(1);
    p0bad.push_back(2);
    p0bad.push_back(1);
    MLK_CHECK(!s0.domain.containsPoint(p0bad));
    SmallVector<int64_t, 8> p1;
    p1.push_back(3);
    p1.push_back(4);
    p1.push_back(2);
    MLK_CHECK(s1.domain.containsPoint(p1));

    // Access map: A[i,k] at (i=1, j=2, k=2) → flat = 1*K + 2 = 5.
    SmallVector<int64_t, 8> at;
    at.push_back(1);
    at.push_back(2);
    at.push_back(2);
    auto flatA = s1.accesses[0].flatIndex.evalAt(at);
    MLK_CHECK(flatA.has_value());
    MLK_CHECK_EQ((*flatA)[0], kTestK * 1 + 2);
    // B[k,j] at the same point → 2*N + 2 = 12.
    auto flatB = s1.accesses[1].flatIndex.evalAt(at);
    MLK_CHECK(flatB.has_value());
    MLK_CHECK_EQ((*flatB)[0], kTestN * 2 + 2);
    // C[i,j] → N + 2 = 7.
    auto flatC = s1.accesses[2].flatIndex.evalAt(at);
    MLK_CHECK(flatC.has_value());
    MLK_CHECK_EQ((*flatC)[0], kTestN * 1 + 2);
}

MLK_TEST(poly, scop_extract_rejects_call) {
    SymbolTable symbols;
    KernelModule km = buildGemmKernel(symbols);
    KernelNode call;
    call.op = mlk::KernelOp::Call;
    (void)km.addNode(call);
    auto scop = extractScop(km, symbols);
    MLK_CHECK(!scop.has_value());
    if (!scop.has_value()) {
        MLK_CHECK(scop.error().code == mlk::ErrorCode::UnsupportedCapability);
    }
}

MLK_TEST(poly, scop_extract_rejects_strided_loop) {
    SymbolTable symbols;
    KernelModule km = buildGemmKernel(symbols);
    // Node layout: [compute1=0, store1=1, compute0=2, store0=3, kLoop=4,
    // jLoop=5, iLoop=6] (see builder). The k loop is node id 4.
    km.nodes[4].step = 2;
    auto scop = extractScop(km, symbols);
    MLK_CHECK(!scop.has_value());
}

MLK_TEST(poly, scop_extract_legacy_1d) {
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer x;
    x.name = symbols.intern("x");
    x.dims = SmallVector<int64_t, 4>{8};
    x.isInput = true;
    KernelBuffer y;
    y.name = symbols.intern("y");
    y.dims = SmallVector<int64_t, 4>{8};
    y.isOutput = true;
    const uint32_t bx = km.addBuffer(x);
    const uint32_t by = km.addBuffer(y);
    KernelNode compute;
    compute.op = mlk::KernelOp::Compute;
    KernelExpr e;
    e.op = mlk::MathOp::Mul;
    e.a.kind = KernelOperand::Kind::ElemA;
    e.b.kind = KernelOperand::Kind::ElemA;  // x[i]^2 via same buffer
    compute.exprs.push_back(e);
    compute.bufferA = bx;
    KernelNode store;
    store.op = mlk::KernelOp::Store;
    store.bufferOut = by;
    KernelNode loop;
    loop.op = mlk::KernelOp::Loop;
    loop.var = symbols.intern("i");
    loop.begin = 0;
    loop.end = 8;
    {
        uint32_t cid = km.addNode(compute);
        uint32_t sid = km.addNode(store);
        loop.children.push_back(cid);
        loop.children.push_back(sid);
    }
    (void)km.addNode(loop);

    auto scop = extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;
    MLK_CHECK_EQ(scop->depth, 1);
    MLK_CHECK_EQ(scop->statements.size(), 1);
    const Statement& s = scop->statements[0];
    // Both operand slots reference x[i] (ElemA twice): two reads + write.
    MLK_CHECK_EQ(s.accesses.size(), 3);
    MLK_CHECK(!s.accesses[0].isWrite);
    MLK_CHECK(!s.accesses[1].isWrite);
    MLK_CHECK(s.accesses[2].isWrite);
    // Read map: flat = i.
    SmallVector<int64_t, 8> pt;
    pt.push_back(3);
    auto flat = s.accesses[0].flatIndex.evalAt(pt);
    MLK_CHECK(flat.has_value());
    MLK_CHECK_EQ((*flat)[0], 3);
    // Write map: flat = i.
    auto wflat = s.accesses[2].flatIndex.evalAt(pt);
    MLK_CHECK(wflat.has_value());
    MLK_CHECK_EQ((*wflat)[0], 3);
}

MLK_TEST(poly, scop_extract_rejects_unpaired_compute) {
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer y;
    y.name = symbols.intern("y");
    y.dims = SmallVector<int64_t, 4>{8};
    y.isOutput = true;
    const uint32_t by = km.addBuffer(y);
    KernelNode compute;
    compute.op = mlk::KernelOp::Compute;
    KernelExpr e;
    e.op = mlk::MathOp::Neg;
    e.a.kind = KernelOperand::Kind::Const;
    e.a.constValue = 1.0;
    compute.exprs.push_back(e);
    KernelNode store;
    store.op = mlk::KernelOp::Store;
    store.bufferOut = by;
    KernelNode loop;
    loop.op = mlk::KernelOp::Loop;
    loop.var = symbols.intern("i");
    loop.begin = 0;
    loop.end = 8;
    {
        uint32_t cid = km.addNode(compute);
        uint32_t sid = km.addNode(store);
        // Wrong order: Store BEFORE Compute → pairing fails.
        loop.children.push_back(sid);
        loop.children.push_back(cid);
    }
    (void)km.addNode(loop);
    auto scop = extractScop(km, symbols);
    MLK_CHECK(!scop.has_value());
}


MLK_TEST(poly, dependence_gemm_chain) {
    SymbolTable symbols;
    KernelModule km = buildGemmKernel(symbols);
    auto scop = extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;
    auto deps = mlk::poly::computeDependences(*scop);
    MLK_CHECK(deps.has_value());
    if (!deps.has_value()) return;
    // Expected dependences (accesses: S0=[write C]; S1=[A, B, write C,
    // read C-implicit]):
    //   S0->S1 RAW  (init feeds the accumulator chain)
    //   S0->S1 WAW  (init vs accumulate)
    //   S1->S1 RAW  (accumulator chain, k' > k)
    //   S1->S1 WAW  (accumulator chain, k' > k)
    //   S1->S1 WAR  (read at k, write at k', k' > k)
    //   No dependences touch A or B.
    int rawCross = 0, wawCross = 0, rawIntra = 0, wawIntra = 0, warIntra = 0;
    for (const auto& d : *deps) {
        MLK_CHECK(mlk::poly::dependenceLive(d));
        if (d.srcStmt == 0 && d.dstStmt == 1) {
            if (d.kind == mlk::poly::DepKind::Raw) ++rawCross;
            if (d.kind == mlk::poly::DepKind::Waw) ++wawCross;
            // Tie: same C element → i == i', j == j' (source pins k == 0).
            SmallVector<int64_t, 8> inst;
            // dims: (i, j, k, i', j', k')
            const int64_t vi[6] = {1, 2, 0, 1, 2, 2};
            for (const int64_t v : vi) inst.push_back(v);
            MLK_CHECK(d.relation.containsPoint(inst));
        } else if (d.srcStmt == 1 && d.dstStmt == 1) {
            if (d.kind == mlk::poly::DepKind::Raw) ++rawIntra;
            if (d.kind == mlk::poly::DepKind::Waw) ++wawIntra;
            if (d.kind == mlk::poly::DepKind::War) ++warIntra;
            // Distinct-instance gate: (k'=k) is NOT a dependence.
            SmallVector<int64_t, 8> same;
            const int64_t vs[6] = {1, 2, 1, 1, 2, 1};
            for (const int64_t v : vs) same.push_back(v);
            MLK_CHECK(!d.relation.containsPoint(same));
            // (k'=k+1) IS a dependence.
            SmallVector<int64_t, 8> next;
            const int64_t vn[6] = {1, 2, 1, 1, 2, 2};
            for (const int64_t v : vn) next.push_back(v);
            MLK_CHECK(d.relation.containsPoint(next));
        }
    }
    MLK_CHECK_EQ(rawCross, 1);
    MLK_CHECK_EQ(wawCross, 1);
    MLK_CHECK_EQ(rawIntra, 1);
    MLK_CHECK_EQ(wawIntra, 1);
    MLK_CHECK_EQ(warIntra, 1);
}

MLK_TEST(poly, dependence_free_elementwise) {
    // y[i] = 2 * x[i]: no dependences at all.
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer x;
    x.name = symbols.intern("x");
    x.dims = SmallVector<int64_t, 4>{8};
    x.isInput = true;
    KernelBuffer y;
    y.name = symbols.intern("y");
    y.dims = SmallVector<int64_t, 4>{8};
    y.isOutput = true;
    const uint32_t bx = km.addBuffer(x);
    const uint32_t by = km.addBuffer(y);
    KernelNode compute;
    compute.op = mlk::KernelOp::Compute;
    KernelExpr e;
    e.op = mlk::MathOp::Mul;
    e.a.kind = mlk::KernelOperand::Kind::ElemA;
    KernelExpr two;
    two.op = mlk::MathOp::Mul;
    two.a.kind = mlk::KernelOperand::Kind::Const;
    two.a.constValue = 2.0;
    two.b.kind = mlk::KernelOperand::Kind::Temp;
    two.b.index = 0;
    compute.exprs.push_back(e);
    compute.exprs.push_back(two);
    compute.bufferA = bx;
    KernelNode store;
    store.op = mlk::KernelOp::Store;
    store.bufferOut = by;
    KernelNode loop;
    loop.op = mlk::KernelOp::Loop;
    loop.var = symbols.intern("i");
    loop.begin = 0;
    loop.end = 8;
    {
        uint32_t cid = km.addNode(compute);
        uint32_t sid = km.addNode(store);
        loop.children.push_back(cid);
        loop.children.push_back(sid);
    }
    (void)km.addNode(loop);
    auto scop = mlk::poly::extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;
    auto deps = mlk::poly::computeDependences(*scop);
    MLK_CHECK(deps.has_value());
    if (deps.has_value()) {
        // x/y are distinct buffers; a[i] != a[j] for a write... the only
        // same-buffer pairs are (read x, read x) = RAR (excluded) and
        // (write y, write y) — same instance only → lexGreater empties it.
        MLK_CHECK_EQ(deps->size(), 0);
    }
}

MLK_TEST(poly, dependence_reduction_chain) {
    // y[j] += x[i][j] (accumulate store, 2-D): the accumulator chain gives
    // RAW/WAW/WAR S->S with j' == j and i' > i (reduction-order gate).
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer x;
    x.name = symbols.intern("x");
    x.dims = SmallVector<int64_t, 4>{4, 6};
    x.isInput = true;
    KernelBuffer y;
    y.name = symbols.intern("y");
    y.dims = SmallVector<int64_t, 4>{6};
    y.isOutput = true;
    const uint32_t bx = km.addBuffer(x);
    const uint32_t by = km.addBuffer(y);
    KernelNode compute;
    compute.op = mlk::KernelOp::Compute;
    KernelExpr e;
    e.op = mlk::MathOp::Add;
    e.a.kind = mlk::KernelOperand::Kind::ElemIdx;
    e.a.index = static_cast<int64_t>(bx);
    e.a.idxCoeffs = SmallVector<int64_t, 4>{6, 1};  // x[i*K + j], K = 6
    compute.exprs.push_back(e);
    KernelNode store;
    store.op = mlk::KernelOp::Store;
    store.bufferOut = by;
    // y[j]: one coefficient per enclosing loop var (i, j) → y flat = j.
    store.outIndexCoeffs = SmallVector<int64_t, 4>{0, 1};
    store.accumulate = true;
    KernelNode jLoop;
    jLoop.op = mlk::KernelOp::Loop;
    jLoop.var = symbols.intern("j");
    jLoop.begin = 0;
    jLoop.end = 6;
    {
        uint32_t cid = km.addNode(compute);
        uint32_t sid = km.addNode(store);
        jLoop.children.push_back(cid);
        jLoop.children.push_back(sid);
    }
    KernelNode iLoop;
    iLoop.op = mlk::KernelOp::Loop;
    iLoop.var = symbols.intern("i");
    iLoop.begin = 0;
    iLoop.end = 4;
    {
        uint32_t jid = km.addNode(jLoop);
        iLoop.children.push_back(jid);
    }
    (void)km.addNode(iLoop);

    auto scop = mlk::poly::extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;
    MLK_CHECK_EQ(scop->statements.size(), 1);
    // accumulate -> implicit read: [read x, write y, read y]
    MLK_CHECK_EQ(scop->statements[0].accesses.size(), 3);
    auto deps = mlk::poly::computeDependences(*scop);
    MLK_CHECK(deps.has_value());
    if (!deps.has_value()) return;
    int raw = 0, waw = 0, war = 0;
    for (const auto& d : *deps) {
        if (d.kind == mlk::poly::DepKind::Raw) ++raw;
        if (d.kind == mlk::poly::DepKind::Waw) ++waw;
        if (d.kind == mlk::poly::DepKind::War) ++war;
        // Forward (i=1, j=2) -> (i=3, j=2): the accumulator chain.
        SmallVector<int64_t, 8> fwd;
        const int64_t vf[4] = {1, 2, 3, 2};
        for (const int64_t v : vf) fwd.push_back(v);
        MLK_CHECK(d.relation.containsPoint(fwd));
        // Backward is not a dependence.
        SmallVector<int64_t, 8> back;
        const int64_t vb[4] = {3, 2, 1, 2};
        for (const int64_t v : vb) back.push_back(v);
        MLK_CHECK(!d.relation.containsPoint(back));
        // Same j, same i (same instance) is excluded.
        SmallVector<int64_t, 8> same;
        const int64_t vs[4] = {1, 2, 1, 2};
        for (const int64_t v : vs) same.push_back(v);
        MLK_CHECK(!d.relation.containsPoint(same));
    }
    MLK_CHECK_EQ(raw, 1);
    MLK_CHECK_EQ(waw, 1);
    MLK_CHECK_EQ(war, 1);
}


MLK_TEST(poly, lp_simple_bounded) {
    // minimize x1 + 2*x2 s.t. x1 + x2 == 6, -x1 >= -3 → x1 = 3, x2 = 3.
    mlk::poly::LinearProgram lp;
    lp.nVars = 2;
    SmallVector<mlk::poly::Rational, 8> eq;
    eq.push_back(mlk::poly::Rational{1, 1});
    eq.push_back(mlk::poly::Rational{1, 1});
    lp.eqRows.push_back(std::move(eq));
    lp.eqRhs.push_back(mlk::poly::Rational{6, 1});
    SmallVector<mlk::poly::Rational, 8> ge;
    ge.push_back(mlk::poly::Rational{-1, 1});
    ge.push_back(mlk::poly::Rational{0, 1});
    lp.geRows.push_back(std::move(ge));
    lp.geRhs.push_back(mlk::poly::Rational{-3, 1});
    lp.objective.push_back(mlk::poly::Rational{1, 1});
    lp.objective.push_back(mlk::poly::Rational{2, 1});
    auto sol = mlk::poly::solveLp(lp);
    MLK_CHECK(sol.has_value());
    if (!sol.has_value()) return;
    MLK_CHECK(sol->status == mlk::poly::LpStatus::Optimal);
    MLK_CHECK(sol->objective.num == 9);
    MLK_CHECK(sol->x.size() == 2);
    if (sol->x.size() == 2) {
        MLK_CHECK(sol->x[0].num == 3 && sol->x[0].den == 1);
        MLK_CHECK(sol->x[1].num == 3 && sol->x[1].den == 1);
    }
}

MLK_TEST(poly, lp_infeasible_and_rational) {
    // Infeasible: x >= 1 and x <= 0.
    mlk::poly::LinearProgram bad;
    bad.nVars = 1;
    {
        SmallVector<mlk::poly::Rational, 8> g1;
        g1.push_back(mlk::poly::Rational{1, 1});
        bad.geRows.push_back(std::move(g1));
        bad.geRhs.push_back(mlk::poly::Rational{1, 1});
        SmallVector<mlk::poly::Rational, 8> g2;
        g2.push_back(mlk::poly::Rational{-1, 1});
        bad.geRows.push_back(std::move(g2));
        bad.geRhs.push_back(mlk::poly::Rational{0, 1});
    }
    auto s1 = mlk::poly::solveLp(bad);
    MLK_CHECK(s1.has_value());
    MLK_CHECK(s1->status == mlk::poly::LpStatus::Infeasible);

    // Rational optimum: minimize x/2 s.t. 2x >= 1 → 1/4.
    mlk::poly::LinearProgram lp;
    lp.nVars = 1;
    SmallVector<mlk::poly::Rational, 8> g;
    g.push_back(mlk::poly::Rational{2, 1});
    lp.geRows.push_back(std::move(g));
    lp.geRhs.push_back(mlk::poly::Rational{1, 1});
    lp.objective.push_back(mlk::poly::Rational{1, 2});
    auto s2 = mlk::poly::solveLp(lp);
    MLK_CHECK(s2.has_value());
    MLK_CHECK(s2->status == mlk::poly::LpStatus::Optimal);
    MLK_CHECK(s2->objective.num == 1 && s2->objective.den == 4);
}

MLK_TEST(poly, pluto_gemm_legal_schedule) {
    SymbolTable symbols;
    KernelModule km = buildGemmKernel(symbols);
    auto scop = mlk::poly::extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;
    auto deps = mlk::poly::computeDependences(*scop);
    MLK_CHECK(deps.has_value());
    if (!deps.has_value()) return;
    auto sched = mlk::poly::computePlutoSchedule(*scop, *deps);
    MLK_CHECK(sched.has_value());
    if (!sched.has_value()) return;
    MLK_CHECK_EQ(sched->nStmts, 2);
    MLK_CHECK_EQ(sched->depth, 3);
    MLK_CHECK(sched->rows.size() >= 1);
    MLK_CHECK_EQ(sched->parallel.size(), sched->rows.size());
    // Legality: every dependence preserved (the core contract).
    auto legal = mlk::poly::verifyScheduleLegality(*scop, *deps, *sched);
    MLK_CHECK(legal.has_value());
    MLK_CHECK(legal.has_value() && *legal);
    // Determinism (Rule 53): same input → identical schedule.
    auto sched2 = mlk::poly::computePlutoSchedule(*scop, *deps);
    MLK_CHECK(sched2.has_value());
    if (sched2.has_value()) {
        MLK_CHECK_EQ(sched2->rows.size(), sched->rows.size());
        for (std::size_t r = 0; r < sched->rows.size(); ++r) {
            MLK_CHECK(sched2->rows[r].stmtCoeffs == sched->rows[r].stmtCoeffs);
        }
    }
    // The k-reduction chain must be CARRIED somewhere (some row with
    // parallel=false) — the accumulator order is never fully parallel.
    bool anyNonParallel = false;
    for (const bool p : sched->parallel) anyNonParallel |= !p;
    MLK_CHECK(anyNonParallel);
}

MLK_TEST(poly, pluto_no_deps_totality_row) {
    // y[i] = 2*x[i]: no dependences, but the schedule still covers the
    // varying dim (TOTALITY contract): one identity row on dim 0 so
    // codegen emits the loop and replays every instance exactly once
    // (zero rows would collapse the instance space to a single point).
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer x;
    x.name = symbols.intern("x");
    x.dims = SmallVector<int64_t, 4>{8};
    x.isInput = true;
    KernelBuffer y;
    y.name = symbols.intern("y");
    y.dims = SmallVector<int64_t, 4>{8};
    y.isOutput = true;
    const uint32_t bx = km.addBuffer(x);
    const uint32_t by = km.addBuffer(y);
    KernelNode compute;
    compute.op = mlk::KernelOp::Compute;
    KernelExpr e;
    e.op = mlk::MathOp::Mul;
    e.a.kind = mlk::KernelOperand::Kind::ElemA;
    compute.exprs.push_back(e);
    compute.bufferA = bx;
    KernelNode store;
    store.op = mlk::KernelOp::Store;
    store.bufferOut = by;
    KernelNode loop;
    loop.op = mlk::KernelOp::Loop;
    loop.var = symbols.intern("i");
    loop.begin = 0;
    loop.end = 8;
    {
        uint32_t cid = km.addNode(compute);
        uint32_t sid = km.addNode(store);
        loop.children.push_back(cid);
        loop.children.push_back(sid);
    }
    (void)km.addNode(loop);
    auto scop = mlk::poly::extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;
    auto deps = mlk::poly::computeDependences(*scop);
    MLK_CHECK(deps.has_value());
    MLK_CHECK_EQ(deps->size(), 0);
    auto sched = mlk::poly::computePlutoSchedule(*scop, *deps);
    MLK_CHECK(sched.has_value());
    if (sched.has_value()) {
        // Totality: exactly one row, pivoting the only varying dim,
        // parallel (nothing to carry).
        MLK_CHECK_EQ(sched->rows.size(), std::size_t{1});
        MLK_CHECK_EQ(sched->pivotDim.size(), std::size_t{1});
        if (sched->rows.size() == 1) {
            MLK_CHECK_EQ(sched->pivotDim[0], 0u);
            MLK_CHECK(sched->parallel[0]);
            const auto& cs = sched->rows[0].stmtCoeffs[0];
            MLK_CHECK(cs.size() >= 2 && cs[1] == 1);
        }
        auto legal = mlk::poly::verifyScheduleLegality(*scop, *deps, *sched);
        MLK_CHECK(legal.has_value() && *legal);
    }
}


MLK_TEST(poly, pluto_fusion_emergence) {
    // Producer/consumer chain t[i] = 2*x[i]; y[i] = t[i] + 1: the parallel
    // row search must FUSE both statements into ONE loop over i (a single
    // parallel schedule row — the old identity+separator scheduler emitted
    // extra separator rows), and the fused kernel must execute bit-exact.
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer x;
    x.name = symbols.intern("x");
    x.dims = SmallVector<int64_t, 4>{8};
    x.isInput = true;
    KernelBuffer t;
    t.name = symbols.intern("t");
    t.dims = SmallVector<int64_t, 4>{8};
    // The runtime ABI passes inputs and outputs only (no intermediate
    // allocation yet — roadmap), so the scratch is an output.
    t.isOutput = true;
    KernelBuffer y;
    y.name = symbols.intern("y");
    y.dims = SmallVector<int64_t, 4>{8};
    y.isOutput = true;
    const uint32_t bx = km.addBuffer(x);
    const uint32_t bt = km.addBuffer(t);
    const uint32_t by = km.addBuffer(y);
    const SymbolId vi = symbols.intern("i");

    // S0: t[i] = 2 * x[i].
    KernelNode compute0;
    compute0.op = mlk::KernelOp::Compute;
    KernelExpr mul;
    mul.op = mlk::MathOp::Mul;
    mul.a.kind = KernelOperand::Kind::ElemIdx;
    mul.a.index = static_cast<int64_t>(bx);
    mul.a.idxCoeffs = SmallVector<int64_t, 4>{1};
    mul.b.kind = KernelOperand::Kind::Const;
    mul.b.constValue = 2.0;
    compute0.exprs.push_back(mul);
    compute0.bufferA = bx;
    KernelNode store0;
    store0.op = mlk::KernelOp::Store;
    store0.bufferOut = bt;
    store0.outIndexCoeffs = SmallVector<int64_t, 4>{1};

    // S1: y[i] = t[i] + 1.
    KernelNode compute1;
    compute1.op = mlk::KernelOp::Compute;
    KernelExpr add;
    add.op = mlk::MathOp::Add;
    add.a.kind = KernelOperand::Kind::ElemIdx;
    add.a.index = static_cast<int64_t>(bt);
    add.a.idxCoeffs = SmallVector<int64_t, 4>{1};
    add.b.kind = KernelOperand::Kind::Const;
    add.b.constValue = 1.0;
    compute1.exprs.push_back(add);
    compute1.bufferA = bt;
    KernelNode store1;
    store1.op = mlk::KernelOp::Store;
    store1.bufferOut = by;
    store1.outIndexCoeffs = SmallVector<int64_t, 4>{1};

    KernelNode loop;
    loop.op = mlk::KernelOp::Loop;
    loop.var = vi;
    loop.begin = 0;
    loop.end = 8;
    {
        uint32_t c0 = km.addNode(compute0);
        uint32_t s0 = km.addNode(store0);
        uint32_t c1 = km.addNode(compute1);
        uint32_t s1 = km.addNode(store1);
        loop.children.push_back(c0);
        loop.children.push_back(s0);
        loop.children.push_back(c1);
        loop.children.push_back(s1);
    }
    (void)km.addNode(loop);

    auto scop = mlk::poly::extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;
    auto deps = mlk::poly::computeDependences(*scop);
    MLK_CHECK(deps.has_value());
    if (!deps.has_value()) return;
    MLK_CHECK_EQ(deps->size(), std::size_t{1});  // the t RAW chain
    auto sched = mlk::poly::computePlutoSchedule(*scop, *deps);
    MLK_CHECK(sched.has_value());
    if (!sched.has_value()) return;
    // FUSION: one parallel row, both statements varying on dim 0 with
    // coefficient 1 (the fused loop), zero separators.
    MLK_CHECK_EQ(sched->rows.size(), std::size_t{1});
    if (sched->rows.size() == 1) {
        MLK_CHECK_EQ(sched->pivotDim[0], 0u);
        MLK_CHECK(sched->parallel[0]);
        const auto& c0 = sched->rows[0].stmtCoeffs[0];
        const auto& c1 = sched->rows[0].stmtCoeffs[1];
        MLK_CHECK(c0.size() >= 2 && c1.size() >= 2);
        MLK_CHECK(c0[1] == 1 && c1[1] == 1);
    }
    auto legal = mlk::poly::verifyScheduleLegality(*scop, *deps, *sched);
    MLK_CHECK(legal.has_value() && *legal);

    // Differential runtime: the fused nest executes bit-exact.
    auto tiled = mlk::poly::computeTiling(*scop, *deps, *sched, 4);
    MLK_CHECK(tiled.has_value());
    if (!tiled.has_value()) return;
    auto out = mlk::poly::emitScheduledKernel(*scop, *sched, *tiled, km,
                                              symbols);
    MLK_CHECK(out.has_value());
    if (!out.has_value()) return;
    SmallVector<double, 8> inX(8), scratchT(8, 0.0), outY(8, 0.0);
    for (std::size_t i = 0; i < 8; ++i) {
        inX[i] = static_cast<double>(i) * 0.5;
    }
    mlk::KernelBufferBindings io;
    io.inputs.push_back(inX.data());
    io.outputs.push_back(scratchT.data());  // t: output slot 0
    io.outputs.push_back(outY.data());      // y: output slot 1
    io.elements = 8;
    auto r = mlk::executeKernelOnBuffers(*out, symbols, io, nullptr);
    MLK_CHECK(r.has_value());
    if (!r.has_value()) return;
    for (std::size_t i = 0; i < 8; ++i) {
        const double ref = 2.0 * inX[i] + 1.0;
        MLK_CHECK(outY[i] == ref);  // bit-exact (Rule 43)
    }
}

MLK_TEST(poly, pluto_wavefront_nest) {
    // b[i][j] = b[i-1][j] + b[i][j-1] over i,j in [1,3] (the classic
    // wavefront dependences (1,0) and (0,1), self-fed so the reads have
    // real producers): the parallel search must fail (no zero-distance
    // row exists), the sequential rows carry each dependence exactly
    // once, and the schedule is legal.
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer b;
    b.name = symbols.intern("b");
    b.dims = SmallVector<int64_t, 4>{4, 4};
    b.isInput = true;
    b.isOutput = true;
    const uint32_t bb = km.addBuffer(b);
    const SymbolId vi = symbols.intern("i");
    const SymbolId vj = symbols.intern("j");

    KernelNode compute;
    compute.op = mlk::KernelOp::Compute;
    KernelExpr e1;
    e1.op = mlk::MathOp::Add;
    e1.a.kind = KernelOperand::Kind::ElemIdx;
    e1.a.index = static_cast<int64_t>(bb);
    e1.a.idxCoeffs = SmallVector<int64_t, 4>{4, 1};  // 4i + j
    e1.a.idxOffset = -4;                             // b[i-1][j]
    KernelExpr e2;
    e2.op = mlk::MathOp::Add;
    e2.a.kind = KernelOperand::Kind::ElemIdx;
    e2.a.index = static_cast<int64_t>(bb);
    e2.a.idxCoeffs = SmallVector<int64_t, 4>{4, 1};  // 4i + j
    e2.a.idxOffset = -1;                             // b[i][j-1]
    KernelExpr sum;
    sum.op = mlk::MathOp::Add;
    sum.a.kind = KernelOperand::Kind::Temp;
    sum.a.index = 0;
    sum.b.kind = KernelOperand::Kind::Temp;
    sum.b.index = 1;
    compute.exprs.push_back(e1);
    compute.exprs.push_back(e2);
    compute.exprs.push_back(sum);
    compute.bufferA = bb;
    KernelNode store;
    store.op = mlk::KernelOp::Store;
    store.bufferOut = bb;
    store.outIndexCoeffs = SmallVector<int64_t, 4>{4, 1};

    KernelNode jLoop;
    jLoop.op = mlk::KernelOp::Loop;
    jLoop.var = vj;
    jLoop.begin = 1;
    jLoop.end = 4;
    {
        uint32_t c = km.addNode(compute);
        uint32_t st = km.addNode(store);
        jLoop.children.push_back(c);
        jLoop.children.push_back(st);
    }
    KernelNode iLoop;
    iLoop.op = mlk::KernelOp::Loop;
    iLoop.var = vi;
    iLoop.begin = 1;
    iLoop.end = 4;
    iLoop.children.push_back(km.addNode(jLoop));
    (void)km.addNode(iLoop);

    auto scop = mlk::poly::extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;
    auto deps = mlk::poly::computeDependences(*scop);
    MLK_CHECK(deps.has_value());
    if (!deps.has_value()) return;
    MLK_CHECK(deps->size() >= 2);  // RAW (1,0) and RAW (0,1)
    auto sched = mlk::poly::computePlutoSchedule(*scop, *deps);
    MLK_CHECK(sched.has_value());
    if (!sched.has_value()) return;
    // Both rows carry: [i] resolves (1,0); [j] resolves (0,1) on the
    // i-equal slice. No parallel row exists.
    MLK_CHECK_EQ(sched->rows.size(), std::size_t{2});
    if (sched->rows.size() == 2) {
        MLK_CHECK(!sched->parallel[0]);
        MLK_CHECK(!sched->parallel[1]);
        MLK_CHECK_EQ(sched->pivotDim[0], 0u);
        MLK_CHECK_EQ(sched->pivotDim[1], 1u);
    }
    auto legal = mlk::poly::verifyScheduleLegality(*scop, *deps, *sched);
    MLK_CHECK(legal.has_value() && *legal);
    // Determinism (Rule 53).
    auto sched2 = mlk::poly::computePlutoSchedule(*scop, *deps);
    MLK_CHECK(sched2.has_value());
    if (sched2.has_value()) {
        MLK_CHECK(sched2->rows[0].stmtCoeffs == sched->rows[0].stmtCoeffs);
        MLK_CHECK(sched2->rows[1].stmtCoeffs == sched->rows[1].stmtCoeffs);
    }
}

MLK_TEST(poly, pluto_sor_time_space_parallel) {
    // 1-D SOR over a time dimension: a[t][i] = a[t-1][i-1] + a[t-1][i] +
    // a[t-1][i+1]. Dependences (1,-1),(1,0),(1,+1): the t row is
    // sequential (carries all three); the space row is PARALLEL — the
    // wavefront-parallel structure, derived by the LP (not copied).
    SymbolTable symbols;
    KernelModule km;
    constexpr int64_t kT = 4;
    constexpr int64_t kN = 8;
    KernelBuffer a;
    a.name = symbols.intern("a");
    a.dims = SmallVector<int64_t, 4>{kT, kN};
    a.isInput = true;
    a.isOutput = true;
    const uint32_t ba = km.addBuffer(a);
    const SymbolId vt = symbols.intern("t");
    const SymbolId vi = symbols.intern("i");

    KernelNode compute;
    compute.op = mlk::KernelOp::Compute;
    KernelExpr e1;
    e1.op = mlk::MathOp::Add;
    e1.a.kind = KernelOperand::Kind::ElemIdx;
    e1.a.index = static_cast<int64_t>(ba);
    e1.a.idxCoeffs = SmallVector<int64_t, 4>{kN, 1};  // t*N + i
    e1.a.idxOffset = -kN - 1;                         // a[t-1][i-1]
    KernelExpr e2;
    e2.op = mlk::MathOp::Add;
    e2.a.kind = KernelOperand::Kind::ElemIdx;
    e2.a.index = static_cast<int64_t>(ba);
    e2.a.idxCoeffs = SmallVector<int64_t, 4>{kN, 1};
    e2.a.idxOffset = -kN;                             // a[t-1][i]
    KernelExpr e3;
    e3.op = mlk::MathOp::Add;
    e3.a.kind = KernelOperand::Kind::Temp;
    e3.a.index = 0;
    e3.b.kind = KernelOperand::Kind::ElemIdx;
    e3.b.index = static_cast<int64_t>(ba);
    e3.b.idxCoeffs = SmallVector<int64_t, 4>{kN, 1};
    e3.b.idxOffset = -kN + 1;                         // a[t-1][i+1]
    KernelExpr sum2;
    sum2.op = mlk::MathOp::Add;
    sum2.a.kind = KernelOperand::Kind::Temp;
    sum2.a.index = 2;
    sum2.b.kind = KernelOperand::Kind::Const;
    sum2.b.constValue = 0.0;  // 3-way sum via two Adds + identity tail
    compute.exprs.push_back(e1);
    compute.exprs.push_back(e2);
    compute.exprs.push_back(e3);
    compute.exprs.push_back(sum2);
    compute.bufferA = ba;
    KernelNode store;
    store.op = mlk::KernelOp::Store;
    store.bufferOut = ba;
    store.outIndexCoeffs = SmallVector<int64_t, 4>{kN, 1};

    KernelNode iLoop;
    iLoop.op = mlk::KernelOp::Loop;
    iLoop.var = vi;
    iLoop.begin = 1;
    iLoop.end = kN - 1;
    {
        uint32_t c = km.addNode(compute);
        uint32_t st = km.addNode(store);
        iLoop.children.push_back(c);
        iLoop.children.push_back(st);
    }
    KernelNode tLoop;
    tLoop.op = mlk::KernelOp::Loop;
    tLoop.var = vt;
    tLoop.begin = 1;
    tLoop.end = kT;
    tLoop.children.push_back(km.addNode(iLoop));
    (void)km.addNode(tLoop);

    auto scop = mlk::poly::extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;
    auto deps = mlk::poly::computeDependences(*scop);
    MLK_CHECK(deps.has_value());
    if (!deps.has_value()) return;
    MLK_CHECK(deps->size() >= 3);  // three RAW stencils
    auto sched = mlk::poly::computePlutoSchedule(*scop, *deps);
    MLK_CHECK(sched.has_value());
    if (!sched.has_value()) return;
    MLK_CHECK_EQ(sched->rows.size(), std::size_t{2});
    if (sched->rows.size() == 2) {
        MLK_CHECK_EQ(sched->pivotDim[0], 0u);  // t first: carries all
        MLK_CHECK_EQ(sched->pivotDim[1], 1u);  // i: space loop
        MLK_CHECK(!sched->parallel[0]);
        MLK_CHECK(sched->parallel[1]);  // SPACE PARALLELISM
        MLK_CHECK(sched->vectorizable[1]);
    }
    auto legal = mlk::poly::verifyScheduleLegality(*scop, *deps, *sched);
    MLK_CHECK(legal.has_value() && *legal);
}

MLK_TEST(poly, tiling_gemm_band) {
    SymbolTable symbols;
    KernelModule km = buildGemmKernel(symbols);
    auto scop = mlk::poly::extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;
    auto deps = mlk::poly::computeDependences(*scop);
    MLK_CHECK(deps.has_value());
    if (!deps.has_value()) return;
    auto sched = mlk::poly::computePlutoSchedule(*scop, *deps);
    MLK_CHECK(sched.has_value());
    if (!sched.has_value()) return;
    // Tiling consistency: the band is a prefix of the schedule rows; every
    // band row carries a tile size; dims carried by the reduction chain
    // are never parallel-marked.
    auto tiled = mlk::poly::computeTiling(*scop, *deps, *sched, 32);
    MLK_CHECK(tiled.has_value());
    if (!tiled.has_value()) return;
    MLK_CHECK(tiled->bandEnd <= sched->rows.size());
    MLK_CHECK_EQ(tiled->tileSizes.size(),
                 static_cast<std::size_t>(tiled->bandEnd));
    MLK_CHECK(tiled->tiled == (tiled->bandEnd > 0));
    for (const int64_t t : tiled->tileSizes) MLK_CHECK_EQ(t, 32);
    bool anyNonParallel = false;
    for (const bool p : sched->parallel) anyNonParallel |= !p;
    MLK_CHECK(anyNonParallel);
}

MLK_TEST(poly, tiling_no_deps_no_band) {
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer x;
    x.name = symbols.intern("x");
    x.dims = SmallVector<int64_t, 4>{8};
    x.isInput = true;
    KernelBuffer y;
    y.name = symbols.intern("y");
    y.dims = SmallVector<int64_t, 4>{8};
    y.isOutput = true;
    const uint32_t bx = km.addBuffer(x);
    const uint32_t by = km.addBuffer(y);
    KernelNode compute;
    compute.op = mlk::KernelOp::Compute;
    KernelExpr e;
    e.op = mlk::MathOp::Mul;
    e.a.kind = mlk::KernelOperand::Kind::ElemA;
    compute.exprs.push_back(e);
    compute.bufferA = bx;
    KernelNode store;
    store.op = mlk::KernelOp::Store;
    store.bufferOut = by;
    KernelNode loop;
    loop.op = mlk::KernelOp::Loop;
    loop.var = symbols.intern("i");
    loop.begin = 0;
    loop.end = 8;
    {
        uint32_t cid = km.addNode(compute);
        uint32_t sid = km.addNode(store);
        loop.children.push_back(cid);
        loop.children.push_back(sid);
    }
    (void)km.addNode(loop);
    auto scop = mlk::poly::extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;
    auto deps = mlk::poly::computeDependences(*scop);
    MLK_CHECK(deps.has_value());
    auto sched = mlk::poly::computePlutoSchedule(*scop, *deps);
    MLK_CHECK(sched.has_value());
    if (!sched.has_value()) return;
    auto tiled = mlk::poly::computeTiling(*scop, *deps, *sched, 32);
    MLK_CHECK(tiled.has_value());
    if (tiled.has_value()) {
        MLK_CHECK(!tiled->tiled);  // zero rows: nothing to tile
        MLK_CHECK_EQ(tiled->bandEnd, 0);
    }
}


MLK_TEST(poly, codegen_gemm_fused_nest) {
    SymbolTable symbols;
    KernelModule km = buildGemmKernel(symbols);
    auto scop = mlk::poly::extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;
    auto deps = mlk::poly::computeDependences(*scop);
    MLK_CHECK(deps.has_value());
    if (!deps.has_value()) return;
    auto sched = mlk::poly::computePlutoSchedule(*scop, *deps);
    MLK_CHECK(sched.has_value());
    if (!sched.has_value()) return;
    // Untiled emission for the structure check (tiled form is covered by
    // the next test).
    mlk::poly::TiledInfo untiled;
    auto out = mlk::poly::emitScheduledKernel(*scop, *sched, untiled, km,
                                              symbols);
    MLK_CHECK(out.has_value());
    if (!out.has_value()) return;
    // Expected tree (the fused schedule [i, k, j]: the order search
    // picked the innermost-j SIMD stride fit; the k row carries the
    // reduction chain; the init statement is row-constant at the k row
    // and re-enters via the PIECEWISE SPLIT — the k range is cut at the
    // init's folded value 0, so the init pair emits unguarded in the
    // singleton segment and the hot k >= 1 segment carries no branch):
    //   for i(par) { for k in [0,1)  { for j(par, vector) {
    //     C[i,j] = 0; C[i,j] += A[i,k] * B[k,j] } }
    //                for k in [1,K) { for j(par, vector) {
    //     C[i,j] += A[i,k] * B[k,j] } } }
    const KernelModule& km2 = *out;
    MLK_CHECK_EQ(km2.buffers.size(), km.buffers.size());
    MLK_CHECK(!km2.nodes.empty());
    // No Guard nodes anywhere: the split realized every re-entry
    // guard structurally (Rule 78 shape, checked by scan below).
    for (const auto& n : km2.nodes) {
        MLK_CHECK(n.op != mlk::KernelOp::Guard);
    }
    // Find the root (never referenced as a child).
    uint32_t root = 0xFFFFFFFFu;
    {
        SmallVector<bool, 8> ref(km2.nodes.size(), false);
        for (const auto& n : km2.nodes) {
            for (const uint32_t c : n.children) {
                if (c < ref.size()) ref[c] = true;
            }
        }
        for (uint32_t i = 0; i < km2.nodes.size(); ++i) {
            if (!ref[i] && km2.nodes[i].op == mlk::KernelOp::Loop) {
                root = i;
                break;
            }
        }
    }
    MLK_CHECK(root != 0xFFFFFFFFu);
    if (root == 0xFFFFFFFFu) return;
    // i loop: the outermost parallel row (zero-distance).
    const KernelNode& iLoop = km2.nodes[root];
    MLK_CHECK_EQ(iLoop.begin, 0);
    MLK_CHECK_EQ(iLoop.end, kTestM);
    MLK_CHECK(iLoop.parallel);
    // Two sibling k segments: the init's singleton [0, 0] and the hot
    // [1, K) body.
    MLK_CHECK_EQ(iLoop.children.size(), std::size_t{2});
    // k segment 0: singleton (end = 1), carries the init pair + acc.
    const KernelNode& k0 = km2.nodes[iLoop.children[0]];
    MLK_CHECK_EQ(k0.op, mlk::KernelOp::Loop);
    MLK_CHECK_EQ(k0.begin, 0);
    MLK_CHECK_EQ(k0.end, 1);
    MLK_CHECK(!k0.parallel);
    MLK_CHECK_EQ(k0.children.size(), std::size_t{1});
    const KernelNode& j0 = km2.nodes[k0.children[0]];
    MLK_CHECK_EQ(j0.op, mlk::KernelOp::Loop);
    MLK_CHECK_EQ(j0.end, kTestN);
    MLK_CHECK(j0.parallel);
    MLK_CHECK(j0.vectorHint);
    // Segment 0's j body: [init compute, init store, acc compute,
    // acc store] — the origOrder tie-break keeps init before acc at the
    // fully-tied schedule vectors, exactly like the guarded form.
    MLK_CHECK_EQ(j0.children.size(), std::size_t{4});
    const KernelNode& initC = km2.nodes[j0.children[0]];
    const KernelNode& initS = km2.nodes[j0.children[1]];
    const KernelNode& accC0 = km2.nodes[j0.children[2]];
    const KernelNode& accS0 = km2.nodes[j0.children[3]];
    MLK_CHECK_EQ(initC.op, mlk::KernelOp::Compute);
    MLK_CHECK_EQ(initS.op, mlk::KernelOp::Store);
    MLK_CHECK(!initS.accumulate);
    MLK_CHECK_EQ(accC0.op, mlk::KernelOp::Compute);
    MLK_CHECK_EQ(accS0.accumulate, true);
    // k segment 1: the hot range [1, K) — acc only, no branch.
    const KernelNode& k1 = km2.nodes[iLoop.children[1]];
    MLK_CHECK_EQ(k1.op, mlk::KernelOp::Loop);
    MLK_CHECK_EQ(k1.begin, 1);
    MLK_CHECK_EQ(k1.end, kTestK);
    MLK_CHECK(!k1.parallel);
    MLK_CHECK_EQ(k1.children.size(), std::size_t{1});
    const KernelNode& j1 = km2.nodes[k1.children[0]];
    MLK_CHECK_EQ(j1.op, mlk::KernelOp::Loop);
    MLK_CHECK_EQ(j1.end, kTestN);
    MLK_CHECK(j1.parallel);
    MLK_CHECK(j1.vectorHint);
    MLK_CHECK_EQ(j1.children.size(), std::size_t{2});
    const KernelNode& accC = km2.nodes[j1.children[0]];
    const KernelNode& accS = km2.nodes[j1.children[1]];
    MLK_CHECK_EQ(accC.op, mlk::KernelOp::Compute);
    MLK_CHECK_EQ(accS.accumulate, true);
    // Acc chain: [A load, B load, mul] with stack-mapped operands:
    // stack = [i, k, j] -> A coeffs [K, 1, 0] (unit stride along k).
    MLK_CHECK_EQ(accC.exprs.size(), 3);
    MLK_CHECK(accC.exprs[0].a.kind == mlk::KernelOperand::Kind::ElemIdx);
    MLK_CHECK_EQ(accC.exprs[0].a.idxCoeffs.size(), 3);
    MLK_CHECK_EQ(accC.exprs[0].a.idxCoeffs[0], kTestK);
    MLK_CHECK_EQ(accC.exprs[0].a.idxCoeffs[1], 1);
}

MLK_TEST(poly, codegen_tiled_gemm) {
    SymbolTable symbols;
    KernelModule km = buildGemmKernel(symbols);
    auto scop = mlk::poly::extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    auto deps = mlk::poly::computeDependences(*scop);
    MLK_CHECK(deps.has_value());
    auto sched = mlk::poly::computePlutoSchedule(*scop, *deps);
    MLK_CHECK(sched.has_value());
    if (!sched.has_value()) return;
    // Tile size 2 divides M=4, N=5? N-1=4: (hi+1)=5 % 2 != 0 → the j
    // level bails; tile 2 divides i (4) and k (3? no: 3 % 2 != 0 → bail).
    // Use tile 1: divides everything (identity tiles).
    auto tiled = mlk::poly::computeTiling(*scop, *deps, *sched, 1);
    MLK_CHECK(tiled.has_value());
    if (!tiled.has_value()) return;
    auto out = mlk::poly::emitScheduledKernel(*scop, *sched, *tiled, km,
                                              symbols);
    MLK_CHECK(out.has_value());
    if (!out.has_value()) return;
    // Tile size 1: tile loop == point loop bounds; structure preserved.
    const KernelModule& km2 = *out;
    MLK_CHECK(!km2.nodes.empty());
    auto legal2 = mlk::poly::verifyScheduleLegality(*scop, *deps, *sched);
    MLK_CHECK(legal2.has_value() && *legal2);
}


MLK_TEST(poly, full_chain_baseline_to_transformed) {
    // THE end-to-end milestone: a baseline GEMM kernel (Call node, the
    // lower.to_kernel_ir shape) flows through poly.synth -> scop_detect ->
    // dependence -> schedule -> tile -> codegen and comes out as a legal
    // fused, tiled loop forest.
    SymbolTable symbols;
    mlk::DiagnosticEngine diag;
    mlk::PassContext ctx;
    ctx.symbols = &symbols;
    ctx.diag = &diag;
    ctx.tier = mlk::Tier::Tier2;
    mlk::poly::PolyWorkspace* ws = mlk::poly::createPolyWorkspace();
    ctx.polyWorkspace = ws;
    mlk::MathGraph graph(&symbols);
    mlk::passes::registerAllPasses(symbols);

    // Baseline: buffers + a single Call(MatMul) node.
    KernelModule km;
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = SmallVector<int64_t, 4>{kTestM, kTestK};
    a.isInput = true;
    KernelBuffer b;
    b.name = symbols.intern("B");
    b.dims = SmallVector<int64_t, 4>{kTestK, kTestN};
    b.isInput = true;
    KernelBuffer c;
    c.name = symbols.intern("C");
    c.dims = SmallVector<int64_t, 4>{kTestM, kTestN};
    c.isOutput = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufB = km.addBuffer(b);
    const uint32_t bufC = km.addBuffer(c);
    KernelNode call;
    call.op = mlk::KernelOp::Call;
    call.math = mlk::MathOp::MatMul;
    call.bufferA = bufA;
    call.bufferB = bufB;
    call.bufferOut = bufC;
    (void)km.addNode(call);
    ctx.kernelOut = &km;

    auto passFn = [&](const char* name) -> mlk::Pass* {
        mlk::Pass* p =
            mlk::PassRegistry::instance().byName(symbols.intern(name));
        MLK_CHECK(p != nullptr);
        return p;
    };

    // 1. poly.synth: Call -> init + accumulate nest.
    {
        auto r = passFn("poly.synth")->run(ctx, graph);
        MLK_CHECK(r.has_value());
        MLK_CHECK(r->changed);
    }
    MLK_CHECK_EQ(km.nodes.size(), std::size_t{7});

    // 2. scop_detect.
    {
        auto r = passFn("poly.scop_detect")->run(ctx, graph);
        MLK_CHECK(r.has_value());
        MLK_CHECK(ws->scopValid);
        MLK_CHECK_EQ(ws->scop.statements.size(), 2);
    }
    // 3. dependences.
    {
        auto r = passFn("poly.dependence")->run(ctx, graph);
        MLK_CHECK(r.has_value());
        MLK_CHECK(ws->dependencesValid);
        MLK_CHECK(ws->dependences.size() >= 3);
    }
    // 4. schedule.
    {
        auto r = passFn("poly.schedule")->run(ctx, graph);
        MLK_CHECK(r.has_value());
        MLK_CHECK(ws->scheduleValid);
    }
    // 5. tile.
    {
        auto r = passFn("poly.tile")->run(ctx, graph);
        MLK_CHECK(r.has_value());
        MLK_CHECK(ws->tileValid);
    }
    // 6. codegen.
    {
        auto r = passFn("poly.codegen")->run(ctx, graph);
        MLK_CHECK(r.has_value());
        MLK_CHECK(ws->codegenValid);
        MLK_CHECK(r->changed);
    }
    // Codegen emitted the transformed forest: the fused GEMM nest plus
    // tile/point splits from the default tile knob (>= 7 nodes).
    MLK_CHECK(km.nodes.size() >= std::size_t{7});
    // The output buffer survived with its shape.
    MLK_CHECK_EQ(km.buffers.size(), std::size_t{3});
    MLK_CHECK_EQ(km.buffers[bufC].dims.size(), std::size_t{2});

    mlk::poly::destroyPolyWorkspace(ws);
    (void)bufC;
}


MLK_TEST(poly, runtime_execute_transformed_gemm) {
    // Differential milestone (Rules 43/85/90): run the TRANSFORMED GEMM
    // kernel over dense buffers and compare against a straightforward
    // reference implementation — bit-exact (same accumulation order per
    // output cell: k ascending).
    SymbolTable symbols;
    mlk::DiagnosticEngine diag;
    mlk::PassContext ctx;
    ctx.symbols = &symbols;
    ctx.diag = &diag;
    ctx.tier = mlk::Tier::Tier2;
    mlk::poly::PolyWorkspace* ws = mlk::poly::createPolyWorkspace();
    ctx.polyWorkspace = ws;
    mlk::MathGraph graph(&symbols);
    mlk::passes::registerAllPasses(symbols);

    constexpr int64_t M = 8, K = 6, N = 7;
    KernelModule km;
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = SmallVector<int64_t, 4>{M, K};
    a.isInput = true;
    KernelBuffer b;
    b.name = symbols.intern("B");
    b.dims = SmallVector<int64_t, 4>{K, N};
    b.isInput = true;
    KernelBuffer c;
    c.name = symbols.intern("C");
    c.dims = SmallVector<int64_t, 4>{M, N};
    c.isOutput = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufB = km.addBuffer(b);
    const uint32_t bufC = km.addBuffer(c);
    KernelNode call;
    call.op = mlk::KernelOp::Call;
    call.math = mlk::MathOp::MatMul;
    call.bufferA = bufA;
    call.bufferB = bufB;
    call.bufferOut = bufC;
    (void)km.addNode(call);
    ctx.kernelOut = &km;

    mlk::MathGraph g2(&symbols);
    auto passFn = [&](const char* name) -> mlk::Pass* {
        return mlk::PassRegistry::instance().byName(symbols.intern(name));
    };
    for (const char* name :
         {"poly.synth", "poly.scop_detect", "poly.dependence",
          "poly.schedule", "poly.tile", "poly.codegen", "poly.verify"}) {
        auto r = passFn(name)->run(ctx, g2);
        MLK_CHECK(r.has_value());
    }
    MLK_CHECK(ws->codegenValid);

    // Bind buffers with a deterministic pattern.
    const std::size_t nA = static_cast<std::size_t>(M * K);
    const std::size_t nB = static_cast<std::size_t>(K * N);
    const std::size_t nC = static_cast<std::size_t>(M * N);
    SmallVector<double, 8> bufInA(nA), bufInB(nB), bufOutC(nC, 0.0);
    for (std::size_t i = 0; i < nA; ++i) {
        bufInA[i] = static_cast<double>((i * 7) % 13) * 0.25;
    }
    for (std::size_t i = 0; i < nB; ++i) {
        bufInB[i] = static_cast<double>((i * 5) % 11) * 0.5;
    }
    mlk::KernelBufferBindings io;
    io.inputs.push_back(bufInA.data());
    io.inputs.push_back(bufInB.data());
    io.outputs.push_back(bufOutC.data());
    io.elements = M * K;

    auto r = mlk::executeKernelOnBuffers(km, symbols, io, nullptr);
    MLK_CHECK(r.has_value());
    if (!r.has_value()) {
        mlk::poly::destroyPolyWorkspace(ws);
        return;
    }
    // Reference: C[i][j] = sum_k A[i][k] * B[k][j] (k ascending).
    int mismatches = 0;
    for (int64_t i = 0; i < M && mismatches == 0; ++i) {
        for (int64_t j = 0; j < N && mismatches == 0; ++j) {
            double ref = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                ref += bufInA[static_cast<std::size_t>(i * K + k)] *
                       bufInB[static_cast<std::size_t>(k * N + j)];
            }
            const double got =
                bufOutC[static_cast<std::size_t>(i * N + j)];
            if (std::fabs(ref - got) > 1e-9) ++mismatches;
        }
    }
    MLK_CHECK_EQ(mismatches, 0);
    mlk::poly::destroyPolyWorkspace(ws);
}

MLK_TEST(poly, pluto_interchange_by_stride_fit) {
    // THE permutation milestone: two statements whose accesses are
    // unit-stride along dim 0 (transposed-consistent flat indices
    // j*N + i) and tied together (zero-distance RAW). Every pivot order
    // is legal (nothing is carried), both rows are parallel in every
    // order — the EXACT stride-fit criterion must pick the order with
    // dim 0 INNERMOST (pivotDim [1, 0]): the loop INTERCHANGE the
    // ascending-first scheduler never considered. The transformed nest
    // then executes bit-exact.
    SymbolTable symbols;
    KernelModule km;
    constexpr int64_t NI = 4, NJ = 5;
    KernelBuffer a;
    a.name = symbols.intern("a");
    a.dims = SmallVector<int64_t, 4>{NI, NJ};
    a.isInput = true;  // scratch: bound through io.inputs (read+write)
    KernelBuffer y;
    y.name = symbols.intern("y");
    y.dims = SmallVector<int64_t, 4>{NI, NJ};
    y.isOutput = true;
    const uint32_t ba = km.addBuffer(a);
    const uint32_t by = km.addBuffer(y);
    const SymbolId vi = symbols.intern("i");
    const SymbolId vj = symbols.intern("j");

    // S0: a[j*N + i] = 1.0 (Const-only compute; store stride 1 along i).
    KernelNode compute0;
    compute0.op = mlk::KernelOp::Compute;
    KernelExpr one;
    one.op = mlk::MathOp::Add;
    one.a.kind = mlk::KernelOperand::Kind::Const;
    one.a.constValue = 1.0;
    one.b.kind = mlk::KernelOperand::Kind::Const;
    one.b.constValue = 0.0;
    compute0.exprs.push_back(one);
    KernelNode store0;
    store0.op = mlk::KernelOp::Store;
    store0.bufferOut = ba;
    store0.outIndexCoeffs = SmallVector<int64_t, 4>{1, NI};  // j*N + i

    // S1: y[j*N + i] = a[j*N + i] + 1 (unit stride along i everywhere).
    KernelNode compute1;
    compute1.op = mlk::KernelOp::Compute;
    KernelExpr rd;
    rd.op = mlk::MathOp::Add;
    rd.a.kind = mlk::KernelOperand::Kind::ElemIdx;
    rd.a.index = static_cast<int64_t>(ba);
    rd.a.idxCoeffs = SmallVector<int64_t, 4>{1, NI};
    rd.b.kind = mlk::KernelOperand::Kind::Const;
    rd.b.constValue = 1.0;
    compute1.exprs.push_back(rd);
    compute1.bufferA = ba;
    KernelNode store1;
    store1.op = mlk::KernelOp::Store;
    store1.bufferOut = by;
    store1.outIndexCoeffs = SmallVector<int64_t, 4>{1, NI};

    // Baseline nest (identity): i { j { S0; S1 } }.
    KernelNode jLoop;
    jLoop.op = mlk::KernelOp::Loop;
    jLoop.var = vj;
    jLoop.begin = 0;
    jLoop.end = NJ;
    KernelNode iLoop;
    iLoop.op = mlk::KernelOp::Loop;
    iLoop.var = vi;
    iLoop.begin = 0;
    iLoop.end = NI;
    {
        uint32_t c0 = km.addNode(compute0);
        uint32_t s0 = km.addNode(store0);
        uint32_t c1 = km.addNode(compute1);
        uint32_t s1 = km.addNode(store1);
        jLoop.children.push_back(c0);
        jLoop.children.push_back(s0);
        jLoop.children.push_back(c1);
        jLoop.children.push_back(s1);
        const uint32_t jl = km.addNode(jLoop);
        const uint32_t il = km.addNode(iLoop);
        km.nodes[il].children.push_back(jl);  // i outer, j inner
    }

    auto scop = mlk::poly::extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;
    auto deps = mlk::poly::computeDependences(*scop);
    MLK_CHECK(deps.has_value());
    if (!deps.has_value()) return;
    auto sched = mlk::poly::computePlutoSchedule(*scop, *deps);
    MLK_CHECK(sched.has_value());
    if (!sched.has_value()) return;
    // PERMUTED order: dim 1 (j) outermost, dim 0 (i) innermost. Row 0's
    // parallel mark is conservatively false: the transposed tied dep is
    // integer-exactly zero but its RATIONAL relaxation is not (the LP
    // hull admits fractional flat-index collisions); after the row-0
    // distance==0 refinement the relation forces i1 == i2 exactly and
    // row 1 is proven parallel with a unit-stride SIMD fit — the fit
    // term is what selects this order over the identity.
    MLK_CHECK_EQ(sched->rows.size(), std::size_t{2});
    if (sched->rows.size() == 2) {
        MLK_CHECK_EQ(sched->pivotDim[0], 1u);
        MLK_CHECK_EQ(sched->pivotDim[1], 0u);
        MLK_CHECK(sched->parallel[1]);
        MLK_CHECK(sched->vectorizable[1]);  // unit stride along i
    }
    auto legal = mlk::poly::verifyScheduleLegality(*scop, *deps, *sched);
    MLK_CHECK(legal.has_value() && *legal);

    // Differential runtime: the interchanged nest executes bit-exact.
    mlk::poly::TiledInfo untiled;
    auto out = mlk::poly::emitScheduledKernel(*scop, *sched, untiled, km,
                                              symbols);
    MLK_CHECK(out.has_value());
    if (!out.has_value()) return;
    constexpr std::size_t n = static_cast<std::size_t>(NI * NJ);
    SmallVector<double, 8> bufA(n, 0.0), bufY(n, 0.0);
    mlk::KernelBufferBindings io;
    io.inputs.push_back(bufA.data());
    io.outputs.push_back(bufY.data());
    io.elements = n;
    auto r = mlk::executeKernelOnBuffers(*out, symbols, io, nullptr);
    MLK_CHECK(r.has_value());
    if (!r.has_value()) return;
    for (std::size_t idx = 0; idx < n; ++idx) {
        MLK_CHECK(bufA[idx] == 1.0);  // bit-exact (Rule 43)
        MLK_CHECK(bufY[idx] == 2.0);
    }
}

MLK_TEST(poly, runtime_threaded_parallel_loop_determinism) {
    // Threading milestone: a fused 2-statement kernel over a large
    // parallel outer dim (trip >= kParallelChunkElements). The emitted
    // outer loop carries the scheduler's parallel mark and the walker
    // runs disjoint index chunks on threads (when hardware concurrency
    // allows); the result must equal the sequential reference bit-exact
    // (zero-distance rows: slab-disjoint locations, Rule 43/137).
    SymbolTable symbols;
    KernelModule km;
    constexpr int64_t I = 20000, J = 3;
    KernelBuffer x;
    x.name = symbols.intern("x");
    x.dims = SmallVector<int64_t, 4>{I, J};
    x.isInput = true;
    KernelBuffer t;
    t.name = symbols.intern("t");
    t.dims = SmallVector<int64_t, 4>{I, J};
    // The runtime ABI passes inputs and outputs only (no intermediate
    // allocation yet — roadmap), so the scratch is an output.
    t.isOutput = true;
    KernelBuffer y;
    y.name = symbols.intern("y");
    y.dims = SmallVector<int64_t, 4>{I, J};
    y.isOutput = true;
    const uint32_t bx = km.addBuffer(x);
    const uint32_t bt = km.addBuffer(t);
    const uint32_t by = km.addBuffer(y);
    const SymbolId vi = symbols.intern("i");
    const SymbolId vj = symbols.intern("j");

    // S0: t[i][j] = 2 * x[i][j]; S1: y[i][j] = t[i][j] + 1 (tied RAW).
    KernelNode compute0;
    compute0.op = mlk::KernelOp::Compute;
    KernelExpr mul;
    mul.op = mlk::MathOp::Mul;
    mul.a.kind = mlk::KernelOperand::Kind::ElemIdx;
    mul.a.index = static_cast<int64_t>(bx);
    mul.a.idxCoeffs = SmallVector<int64_t, 4>{J, 1};
    mul.b.kind = mlk::KernelOperand::Kind::Const;
    mul.b.constValue = 2.0;
    compute0.exprs.push_back(mul);
    compute0.bufferA = bx;
    KernelNode store0;
    store0.op = mlk::KernelOp::Store;
    store0.bufferOut = bt;
    store0.outIndexCoeffs = SmallVector<int64_t, 4>{J, 1};
    KernelNode compute1;
    compute1.op = mlk::KernelOp::Compute;
    KernelExpr add;
    add.op = mlk::MathOp::Add;
    add.a.kind = mlk::KernelOperand::Kind::ElemIdx;
    add.a.index = static_cast<int64_t>(bt);
    add.a.idxCoeffs = SmallVector<int64_t, 4>{J, 1};
    add.b.kind = mlk::KernelOperand::Kind::Const;
    add.b.constValue = 1.0;
    compute1.exprs.push_back(add);
    compute1.bufferA = bt;
    KernelNode store1;
    store1.op = mlk::KernelOp::Store;
    store1.bufferOut = by;
    store1.outIndexCoeffs = SmallVector<int64_t, 4>{J, 1};

    KernelNode jLoop;
    jLoop.op = mlk::KernelOp::Loop;
    jLoop.var = vj;
    jLoop.begin = 0;
    jLoop.end = J;
    KernelNode iLoop;
    iLoop.op = mlk::KernelOp::Loop;
    iLoop.var = vi;
    iLoop.begin = 0;
    iLoop.end = I;
    {
        uint32_t c0 = km.addNode(compute0);
        uint32_t s0 = km.addNode(store0);
        uint32_t c1 = km.addNode(compute1);
        uint32_t s1 = km.addNode(store1);
        jLoop.children.push_back(c0);
        jLoop.children.push_back(s0);
        jLoop.children.push_back(c1);
        jLoop.children.push_back(s1);
        const uint32_t jl = km.addNode(jLoop);
        const uint32_t il = km.addNode(iLoop);
        km.nodes[il].children.push_back(jl);  // i outer, j inner
    }

    auto scop = mlk::poly::extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;
    auto deps = mlk::poly::computeDependences(*scop);
    MLK_CHECK(deps.has_value());
    if (!deps.has_value()) return;
    auto sched = mlk::poly::computePlutoSchedule(*scop, *deps);
    MLK_CHECK(sched.has_value());
    if (!sched.has_value()) return;
    mlk::poly::TiledInfo untiled;
    auto out = mlk::poly::emitScheduledKernel(*scop, *sched, untiled, km,
                                              symbols);
    MLK_CHECK(out.has_value());
    if (!out.has_value()) return;
    // The outer loop node carries the parallel + vector marks.
    const KernelModule& km2 = *out;
    uint32_t root = 0xFFFFFFFFu;
    {
        SmallVector<bool, 8> referenced(km2.nodes.size(), false);
        for (const KernelNode& nd : km2.nodes) {
            for (const uint32_t c : nd.children) {
                if (c < referenced.size()) referenced[c] = true;
            }
        }
        for (uint32_t i2 = 0; i2 < km2.nodes.size(); ++i2) {
            if (!referenced[i2]) root = i2;
        }
    }
    MLK_CHECK(root != 0xFFFFFFFFu);
    if (root == 0xFFFFFFFFu) return;
    MLK_CHECK_EQ(km2.nodes[root].op, mlk::KernelOp::Loop);
    MLK_CHECK(km2.nodes[root].parallel);
    // The vector hint sits on the innermost parallel level (the j loop
    // inside the root i loop), the row the SIMD fit was proven for.
    bool innerVectorHint = false;
    for (const uint32_t c : km2.nodes[root].children) {
        if (c < km2.nodes.size() &&
            km2.nodes[c].op == mlk::KernelOp::Loop &&
            km2.nodes[c].vectorHint) {
            innerVectorHint = true;
        }
    }
    MLK_CHECK(innerVectorHint);

    // Execute (threaded when hardware_concurrency >= 2) + reference.
    constexpr std::size_t n = static_cast<std::size_t>(I * J);
    SmallVector<double, 8> bufX(n), bufT(n, 0.0), bufY(n, 0.0);
    for (std::size_t i2 = 0; i2 < n; ++i2) {
        bufX[i2] = static_cast<double>((i2 * 7) % 13) * 0.25;
    }
    mlk::KernelBufferBindings io;
    io.inputs.push_back(bufX.data());
    io.outputs.push_back(bufT.data());
    io.outputs.push_back(bufY.data());
    io.elements = n;
    auto r = mlk::executeKernelOnBuffers(km2, symbols, io, nullptr);
    MLK_CHECK(r.has_value());
    if (!r.has_value()) return;
    for (std::size_t i2 = 0; i2 < n; ++i2) {
        const double ref = 2.0 * bufX[i2] + 1.0;
        MLK_CHECK(bufY[i2] == ref);  // bit-exact under any interleaving
    }
}

MLK_TEST(poly, reducesum_synth_pipeline_bitexact) {
    // ReduceSum synthesis milestone: baseline Call(ReduceSum) ->
    // poly.synth (init + accumulate nests) -> ... -> codegen with the
    // reduction dim carried and the outer dim parallel; both the
    // BASELINE call path and the transformed nest accumulate k in
    // ascending order (Rule 90) -> bit-exact equality.
    SymbolTable symbols;
    mlk::DiagnosticEngine diag;
    mlk::PassContext ctx;
    ctx.symbols = &symbols;
    ctx.diag = &diag;
    ctx.tier = mlk::Tier::Tier2;
    mlk::poly::PolyWorkspace* ws = mlk::poly::createPolyWorkspace();
    ctx.polyWorkspace = ws;
    mlk::MathGraph graph(&symbols);
    mlk::passes::registerAllPasses(symbols);

    constexpr int64_t M = 32, K = 16;
    KernelModule km;
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = SmallVector<int64_t, 4>{M, K};
    a.isInput = true;
    KernelBuffer y;
    y.name = symbols.intern("Y");
    y.dims = SmallVector<int64_t, 4>{M};
    y.isOutput = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufY = km.addBuffer(y);
    KernelNode call;
    call.op = mlk::KernelOp::Call;
    call.math = mlk::MathOp::ReduceSum;
    call.bufferA = bufA;
    call.bufferOut = bufY;
    (void)km.addNode(call);
    ctx.kernelOut = &km;

    // Baseline execution first (Call path, ascending-k accumulation).
    const std::size_t nA = static_cast<std::size_t>(M * K);
    SmallVector<double, 8> bufInA(nA), baseY(static_cast<std::size_t>(M), 0.0);
    for (std::size_t i = 0; i < nA; ++i) {
        bufInA[i] = static_cast<double>((i * 11) % 17) * 0.125;
    }
    {
        mlk::KernelBufferBindings io;
        io.inputs.push_back(bufInA.data());
        io.outputs.push_back(baseY.data());
        io.elements = nA;
        auto br = mlk::executeKernelOnBuffers(km, symbols, io, nullptr);
        MLK_CHECK(br.has_value());
        if (!br.has_value()) {
            mlk::poly::destroyPolyWorkspace(ws);
            return;
        }
    }

    mlk::MathGraph g2(&symbols);
    auto passFn = [&](const char* name) -> mlk::Pass* {
        return mlk::PassRegistry::instance().byName(symbols.intern(name));
    };
    for (const char* name :
         {"poly.synth", "poly.scop_detect", "poly.dependence",
          "poly.schedule", "poly.tile", "poly.codegen", "poly.verify"}) {
        auto r = passFn(name)->run(ctx, g2);
        MLK_CHECK(r.has_value());
    }
    MLK_CHECK(ws->codegenValid);
    // Schedule shape: [i(par), k(carry)] — the reduction dim carries.
    MLK_CHECK_EQ(ws->schedule.rows.size(), std::size_t{2});
    if (ws->schedule.rows.size() == 2) {
        MLK_CHECK_EQ(ws->schedule.pivotDim[0], 0u);
        MLK_CHECK(ws->schedule.parallel[0]);
        MLK_CHECK_EQ(ws->schedule.pivotDim[1], 1u);
        MLK_CHECK(!ws->schedule.parallel[1]);
    }

    // Transformed execution: bit-exact vs BOTH the reference loop and
    // the baseline call.
    SmallVector<double, 8> bufOutY(static_cast<std::size_t>(M), -1.0);
    mlk::KernelBufferBindings io;
    io.inputs.push_back(bufInA.data());
    io.outputs.push_back(bufOutY.data());
    io.elements = nA;
    auto r = mlk::executeKernelOnBuffers(km, symbols, io, nullptr);
    MLK_CHECK(r.has_value());
    if (!r.has_value()) {
        mlk::poly::destroyPolyWorkspace(ws);
        return;
    }
    int mismatches = 0;
    for (int64_t i = 0; i < M && mismatches == 0; ++i) {
        double ref = 0.0;
        for (int64_t k = 0; k < K; ++k) {
            ref += bufInA[static_cast<std::size_t>(i * K + k)];
        }
        const double got = bufOutY[static_cast<std::size_t>(i)];
        const double base = baseY[static_cast<std::size_t>(i)];
        if (got != ref || base != ref) ++mismatches;  // bit-exact (Rule 43)
    }
    MLK_CHECK_EQ(mismatches, 0);
    mlk::poly::destroyPolyWorkspace(ws);
}

// ---------------------------------------------------------------------------
// Round 13: integer-exact marking, CLAST guard execution, multi-part tiling.
// ---------------------------------------------------------------------------

MLK_TEST(poly, pluto_integer_exact_parallel_mark) {
    // The rational hull of a dependence slice can admit fractional
    // distances that no INTEGER instance pair realizes. The parallel
    // mark must follow the integer points — the instances that execute
    // — not the hull: the hand-built relation below holds
    // 2*i_dst - 2*i_src >= 0 and 2*i_src - 2*i_dst + 1 >= 0, i.e.
    // i_dst == i_src on integers but i_dst - i_src = 1/2 on the hull.
    // The identity row is therefore PARALLEL (the old rational
    // max-distance check marked it sequential).
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer y;
    y.name = symbols.intern("y");
    y.dims = SmallVector<int64_t, 4>{8};
    y.isOutput = true;
    const uint32_t by = km.addBuffer(y);
    const SymbolId vi = symbols.intern("i");

    // S0: y[i] = 1 (overwrite).
    KernelNode compute0;
    compute0.op = mlk::KernelOp::Compute;
    KernelExpr one;
    one.op = mlk::MathOp::Add;
    one.a.kind = KernelOperand::Kind::Const;
    one.a.constValue = 1.0;
    compute0.exprs.push_back(one);
    KernelNode store0;
    store0.op = mlk::KernelOp::Store;
    store0.bufferOut = by;
    store0.outIndexCoeffs = SmallVector<int64_t, 4>{1};

    // S1: y[i] = y[i] + 1 (reads the out buffer; dist-0 tie in program
    // order).
    KernelNode compute1;
    compute1.op = mlk::KernelOp::Compute;
    KernelExpr inc;
    inc.op = mlk::MathOp::Add;
    inc.a.kind = KernelOperand::Kind::ElemIdx;
    inc.a.index = static_cast<int64_t>(by);
    inc.a.idxCoeffs = SmallVector<int64_t, 4>{1};
    inc.b.kind = KernelOperand::Kind::Const;
    inc.b.constValue = 1.0;
    compute1.exprs.push_back(inc);
    KernelNode store1;
    store1.op = mlk::KernelOp::Store;
    store1.bufferOut = by;
    store1.outIndexCoeffs = SmallVector<int64_t, 4>{1};

    KernelNode loop;
    loop.op = mlk::KernelOp::Loop;
    loop.var = vi;
    loop.begin = 0;
    loop.end = 8;
    {
        const uint32_t c0 = km.addNode(compute0);
        const uint32_t s0 = km.addNode(store0);
        const uint32_t c1 = km.addNode(compute1);
        const uint32_t s1 = km.addNode(store1);
        loop.children.push_back(c0);
        loop.children.push_back(s0);
        loop.children.push_back(c1);
        loop.children.push_back(s1);
        (void)km.addNode(loop);
    }

    auto scop = mlk::poly::extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;

    // Hand-built dependence over the product space {i_src, i_dst}: both
    // domains in [0, 8), the parity-tight pair constraints. Integer
    // points: i_dst == i_src only; the rational hull adds the fractional
    // offset 1/2.
    mlk::poly::Dependence dep;
    dep.srcStmt = 0;
    dep.dstStmt = 1;
    dep.kind = mlk::poly::DepKind::Raw;
    dep.relation.space = VarSpace{2, 0};
    {
        mlk::poly::Polyhedron p;
        p.space = VarSpace{2, 0};
        const auto row = [](int64_t a, int64_t b, int64_t c) {
            mlk::poly::ConstraintRow r;
            r.isEquality = false;
            r.coeffs = SmallVector<int64_t, 8>{a, b};
            r.constant = c;
            return r;
        };
        p.addRow(row(1, 0, 0));   // i_src >= 0
        p.addRow(row(-1, 0, 7));  // i_src <= 7
        p.addRow(row(0, 1, 0));   // i_dst >= 0
        p.addRow(row(0, -1, 7));  // i_dst <= 7
        p.addRow(row(-2, 2, 0));  // 2 i_dst - 2 i_src >= 0
        p.addRow(row(2, -2, 1));  // 2 i_src - 2 i_dst + 1 >= 0
        dep.relation.disjuncts.push_back(std::move(p));
    }
    SmallVector<mlk::poly::Dependence, 16> deps;
    deps.push_back(std::move(dep));

    auto sched = mlk::poly::computePlutoSchedule(*scop, deps);
    MLK_CHECK(sched.has_value());
    if (!sched.has_value()) return;
    MLK_CHECK_EQ(sched->rows.size(), std::size_t{1});
    if (sched->rows.size() == 1) {
        // THE assertion: integer-exact zero distance on the slice keeps
        // the row parallel (the rational hull alone would mark it
        // sequential).
        MLK_CHECK(sched->parallel[0]);
    }
    auto legal = mlk::poly::verifyScheduleLegality(*scop, deps, *sched);
    MLK_CHECK(legal.has_value() && *legal);

    // Differential execution: the fused parallel loop replays S0 then
    // S1 per instance (origOrder tie-break at the tied vectors) —
    // y == 2 everywhere, bit-exact (Rule 43).
    mlk::poly::TiledInfo untiled;
    auto out = mlk::poly::emitScheduledKernel(*scop, *sched, untiled, km,
                                              symbols);
    MLK_CHECK(out.has_value());
    if (!out.has_value()) return;
    SmallVector<double, 8> bufY(8, 0.0);
    mlk::KernelBufferBindings io;
    io.outputs.push_back(bufY.data());
    io.elements = 8;
    auto r = mlk::executeKernelOnBuffers(*out, symbols, io, nullptr);
    MLK_CHECK(r.has_value());
    if (!r.has_value()) return;
    for (std::size_t i = 0; i < 8; ++i) {
        MLK_CHECK(bufY[i] == 2.0);
    }
}

MLK_TEST(poly, runtime_guard_predicate_execution) {
    // The CLAST-style affine-equality Guard executes its children only
    // where 1*i - 2 == 0: exactly one iteration performs the guarded
    // write; the unguarded sibling writes everywhere.
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer y;
    y.name = symbols.intern("y");
    y.dims = SmallVector<int64_t, 4>{6};
    y.isOutput = true;
    KernelBuffer z;
    z.name = symbols.intern("z");
    z.dims = SmallVector<int64_t, 4>{6};
    z.isOutput = true;
    const uint32_t by = km.addBuffer(y);
    const uint32_t bz = km.addBuffer(z);
    const SymbolId vi = symbols.intern("i");

    KernelNode computeG;
    computeG.op = mlk::KernelOp::Compute;
    KernelExpr five;
    five.op = mlk::MathOp::Add;
    five.a.kind = KernelOperand::Kind::Const;
    five.a.constValue = 5.0;
    computeG.exprs.push_back(five);
    KernelNode storeG;
    storeG.op = mlk::KernelOp::Store;
    storeG.bufferOut = by;
    storeG.outIndexCoeffs = SmallVector<int64_t, 4>{1};

    KernelNode computeZ;
    computeZ.op = mlk::KernelOp::Compute;
    KernelExpr two;
    two.op = mlk::MathOp::Add;
    two.a.kind = KernelOperand::Kind::Const;
    two.a.constValue = 2.0;
    computeZ.exprs.push_back(two);
    KernelNode storeZ;
    storeZ.op = mlk::KernelOp::Store;
    storeZ.bufferOut = bz;
    storeZ.outIndexCoeffs = SmallVector<int64_t, 4>{1};

    KernelNode guard;
    guard.op = mlk::KernelOp::Guard;
    guard.guardCoeffs = SmallVector<int64_t, 4>{1};
    guard.guardOffset = -2;  // i - 2 == 0

    KernelNode loop;
    loop.op = mlk::KernelOp::Loop;
    loop.var = vi;
    loop.begin = 0;
    loop.end = 6;
    {
        const uint32_t cg = km.addNode(computeG);
        const uint32_t sg = km.addNode(storeG);
        guard.children.push_back(cg);
        guard.children.push_back(sg);
        const uint32_t gid = km.addNode(guard);
        const uint32_t cz = km.addNode(computeZ);
        const uint32_t sz = km.addNode(storeZ);
        loop.children.push_back(gid);
        loop.children.push_back(cz);
        loop.children.push_back(sz);
        (void)km.addNode(loop);
    }

    SmallVector<double, 8> bufY(6, 0.0), bufZ(6, 0.0);
    mlk::KernelBufferBindings io;
    io.outputs.push_back(bufY.data());
    io.outputs.push_back(bufZ.data());
    io.elements = 6;
    auto r = mlk::executeKernelOnBuffers(km, symbols, io, nullptr);
    MLK_CHECK(r.has_value());
    if (!r.has_value()) return;
    for (std::size_t i = 0; i < 6; ++i) {
        MLK_CHECK(bufY[i] == (i == 2 ? 5.0 : 0.0));  // guarded write
        MLK_CHECK(bufZ[i] == 2.0);                   // unguarded write
    }
}

MLK_TEST(poly, runtime_tiled_guarded_gemm_bitexact) {
    // Tile size 2 over M=4, K=3, N=5: the k and j levels carry FULL
    // tiles plus a PARTIAL tail. Tile parts are sibling instance ranges
    // — each replays the same statements for its own point range (the
    // emission marks restore between parts). The guarded init (k == 0)
    // fires only inside the k tile containing k == 0. The tiled kernel
    // must match the untiled transformed kernel bit-exact.
    SymbolTable symbols;
    KernelModule km = buildGemmKernel(symbols);
    auto scop = mlk::poly::extractScop(km, symbols);
    MLK_CHECK(scop.has_value());
    if (!scop.has_value()) return;
    auto deps = mlk::poly::computeDependences(*scop);
    MLK_CHECK(deps.has_value());
    if (!deps.has_value()) return;
    auto sched = mlk::poly::computePlutoSchedule(*scop, *deps);
    MLK_CHECK(sched.has_value());
    if (!sched.has_value()) return;

    mlk::poly::TiledInfo untiled;
    auto flat =
        mlk::poly::emitScheduledKernel(*scop, *sched, untiled, km, symbols);
    MLK_CHECK(flat.has_value());
    auto tiled = mlk::poly::computeTiling(*scop, *deps, *sched, 2);
    MLK_CHECK(tiled.has_value());
    if (!tiled.has_value()) return;
    MLK_CHECK(tiled->tiled);
    auto tiledKm =
        mlk::poly::emitScheduledKernel(*scop, *sched, *tiled, km, symbols);
    MLK_CHECK(tiledKm.has_value());
    if (!flat.has_value() || !tiledKm.has_value()) return;

    // Deterministic inputs.
    constexpr std::size_t nA = static_cast<std::size_t>(kTestM * kTestK);
    constexpr std::size_t nB = static_cast<std::size_t>(kTestK * kTestN);
    constexpr std::size_t nC = static_cast<std::size_t>(kTestM * kTestN);
    SmallVector<double, 8> bufA(nA), bufB(nB);
    for (std::size_t i = 0; i < nA; ++i) {
        bufA[i] = static_cast<double>(i % 7) * 0.25;
    }
    for (std::size_t i = 0; i < nB; ++i) {
        bufB[i] = static_cast<double>(i % 5) * 0.5;
    }
    SmallVector<double, 8> outFlat(nC, 0.0), outTiled(nC, 0.0);
    mlk::KernelBufferBindings ioFlat;
    ioFlat.inputs.push_back(bufA.data());
    ioFlat.inputs.push_back(bufB.data());
    ioFlat.outputs.push_back(outFlat.data());
    ioFlat.elements = nC;
    auto rFlat = mlk::executeKernelOnBuffers(*flat, symbols, ioFlat,
                                             nullptr);
    MLK_CHECK(rFlat.has_value());
    mlk::KernelBufferBindings ioTiled;
    ioTiled.inputs.push_back(bufA.data());
    ioTiled.inputs.push_back(bufB.data());
    ioTiled.outputs.push_back(outTiled.data());
    ioTiled.elements = nC;
    auto rTiled = mlk::executeKernelOnBuffers(*tiledKm, symbols, ioTiled,
                                              nullptr);
    MLK_CHECK(rTiled.has_value());
    if (!rFlat.has_value() || !rTiled.has_value()) return;
    for (std::size_t i = 0; i < nC; ++i) {
        MLK_CHECK(outFlat[i] == outTiled[i]);  // bit-exact (Rule 43)
    }
    // The partial tile actually ran: cells covered only by the tail
    // tiles (j == 4, k == 2) hold the full product sum, not garbage.
    for (int64_t i = 0; i < kTestM; ++i) {
        double ref = 0.0;
        for (int64_t k = 0; k < kTestK; ++k) {
            ref += bufA[static_cast<std::size_t>(i * kTestK + k)] *
                   bufB[static_cast<std::size_t>(k * kTestN + (kTestN - 1))];
        }
        MLK_CHECK(outTiled[static_cast<std::size_t>(i * kTestN +
                                                   (kTestN - 1))] == ref);
    }
}

// ---------------------------------------------------------------------------
// Round 14: native AOT artifacts — the C++ and x86-64 assembly backends.
// The pipeline here is the full compiler loop: polyhedral schedule ->
// codegen -> emitter (C++ | x86-64 AT&T asm) -> OUT-OF-PROCESS toolchain
// (cc -shared; ADR-0003/0006: no in-process machine codegen) -> dlopen ->
// execute over the buffer ABI -> bit-exact comparison against the buffer
// executor (the artifact mirrors walker semantics by construction).
// ---------------------------------------------------------------------------

/// Cold, cached probe: native-artifact tests require an out-of-process
/// toolchain. On a platform without one the tests SKIP honestly (the
/// skip condition itself is the recorded check); every OTHER failure
/// fails the test — no silent green.
bool backendToolchainReady() {
    static const bool ready = [] {
        mlk::BackendDriverConfig cfg;
        return mlk::toolchainAvailable(cfg);
    }();
    return ready;
}

/// The full-chain GEMM module (scop -> dependences -> schedule ->
/// codegen) shared by the native-artifact tests; tileSize 0 = untiled.
[[nodiscard]] mlk::Result<KernelModule> buildScheduledGemm(
    SymbolTable& symbols, const KernelModule& km, const int64_t tileSize) {
    auto scop = mlk::poly::extractScop(km, symbols);
    if (!scop.has_value()) return std::unexpected<mlk::Error>(scop.error());
    auto deps = mlk::poly::computeDependences(*scop);
    if (!deps.has_value()) return std::unexpected<mlk::Error>(deps.error());
    auto sched = mlk::poly::computePlutoSchedule(*scop, *deps);
    if (!sched.has_value()) return std::unexpected<mlk::Error>(sched.error());
    mlk::poly::TiledInfo tiled;
    if (tileSize > 0) {
        auto t = mlk::poly::computeTiling(*scop, *deps, *sched, tileSize);
        if (!t.has_value()) return std::unexpected<mlk::Error>(t.error());
        tiled = *t;
    }
    return mlk::poly::emitScheduledKernel(*scop, *sched, tiled, km,
                                          symbols);
}

MLK_TEST(poly, backend_legacy_elementwise_bitexact) {
    // Legacy 1-D module: out[i] = A[i] * B[i] over a dynamic bound —
    // the historical ABI form (n + fixed scalars tail). Both artifact
    // forms must match the 1-D executor bit-exact.
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.isInput = true;
    KernelBuffer b;
    b.name = symbols.intern("B");
    b.isInput = true;
    KernelBuffer out;
    out.name = symbols.intern("Out");
    out.isOutput = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufB = km.addBuffer(b);
    const uint32_t bufOut = km.addBuffer(out);
    const SymbolId vi = symbols.intern("i");

    KernelNode loop;
    loop.op = mlk::KernelOp::Loop;
    loop.var = vi;
    loop.begin = 0;
    loop.end = mlk::constants::kKernelLoopDynamicBound;  // n
    KernelNode compute;
    compute.op = mlk::KernelOp::Compute;
    compute.math = mlk::MathOp::Mul;  // legacy single-op form
    compute.bufferA = bufA;
    compute.bufferB = bufB;
    KernelNode store;
    store.op = mlk::KernelOp::Store;
    store.bufferOut = bufOut;
    {
        const uint32_t cid = km.addNode(compute);
        const uint32_t sid = km.addNode(store);
        loop.children.push_back(cid);
        loop.children.push_back(sid);
        (void)km.addNode(loop);
    }

    constexpr int64_t n = 16;
    SmallVector<double, 8> dataA(static_cast<std::size_t>(n));
    SmallVector<double, 8> dataB(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < dataA.size(); ++i) {
        dataA[i] = static_cast<double>(i) * 0.5 - 3.0;
        dataB[i] = static_cast<double>((i * 7) % 5) + 1.0;
    }
    SmallVector<double, 8> refW(static_cast<std::size_t>(n), -99.0);
    mlk::KernelBufferBindings ioW;
    ioW.inputs.push_back(dataA.data());
    ioW.inputs.push_back(dataB.data());
    ioW.outputs.push_back(refW.data());
    ioW.elements = n;
    auto rw = mlk::executeKernelOnBuffers(km, symbols, ioW, nullptr);
    MLK_CHECK(rw.has_value());
    if (!rw.has_value()) return;

    MLK_CHECK(backendToolchainReady());
    if (!backendToolchainReady()) return;
    mlk::BackendDriverConfig cfg;

    SmallVector<double, 8> outCpp(static_cast<std::size_t>(n), -99.0);
    SmallVector<double, 8> outAsm(static_cast<std::size_t>(n), -99.0);
    for (const mlk::ArtifactKind kind : {mlk::ArtifactKind::Cpp,
                                         mlk::ArtifactKind::Asm}) {
        auto loaded = mlk::buildKernelArtifact(km, symbols, kind, cfg);
        MLK_CHECK(loaded.has_value());
        if (!loaded.has_value()) {
            std::fprintf(stderr, "  emit/kind=%d: %s\n",
                         static_cast<int>(kind),
                         loaded.error().message.c_str());
            return;
        }
        MLK_CHECK(!loaded->isMultiDim());
        SmallVector<double, 8>& outC =
            kind == mlk::ArtifactKind::Cpp ? outCpp : outAsm;
        mlk::KernelBufferBindings io;
        io.inputs.push_back(dataA.data());
        io.inputs.push_back(dataB.data());
        io.outputs.push_back(outC.data());
        io.elements = n;
        auto r = loaded->run(km, symbols, io);
        MLK_CHECK(r.has_value());
        if (!r.has_value()) {
            std::fprintf(stderr, "  run/kind=%d: %s\n",
                         static_cast<int>(kind),
                         r.error().message.c_str());
            return;
        }
    }
    for (std::size_t i = 0; i < dataA.size(); ++i) {
        // Bit-exact three-way agreement (Rule 43).
        MLK_CHECK(outCpp[i] == refW[i]);
        MLK_CHECK(outAsm[i] == refW[i]);
    }
}

#if defined(__x86_64__) || defined(_M_X64)
MLK_TEST(poly, asm_backend_gemm_full_chain_bitexact) {
    // The flagship loop: baseline GEMM -> polyhedral schedule -> codegen
    // -> x86-64 assembly artifact -> assembled out-of-process -> loaded
    // -> executed -> bit-exact vs BOTH the C++ artifact and the walker.
    SymbolTable symbols;
    KernelModule base = buildGemmKernel(symbols);
    auto mod = buildScheduledGemm(symbols, base, 0);
    MLK_CHECK(mod.has_value());
    if (!mod.has_value()) return;

    constexpr std::size_t nA = static_cast<std::size_t>(kTestM * kTestK);
    constexpr std::size_t nB = static_cast<std::size_t>(kTestK * kTestN);
    constexpr std::size_t nC = static_cast<std::size_t>(kTestM * kTestN);
    SmallVector<double, 8> bufA(nA), bufB(nB);
    for (std::size_t i = 0; i < nA; ++i) {
        bufA[i] = static_cast<double>(i % 7) * 0.25;
    }
    for (std::size_t i = 0; i < nB; ++i) {
        bufB[i] = static_cast<double>(i % 5) * 0.5;
    }
    SmallVector<double, 8> outW(nC, 0.0), outCpp(nC, 0.0),
        outAsm(nC, 0.0);

    auto bind = [&](SmallVector<double, 8>& outC) {
        mlk::KernelBufferBindings io;
        io.inputs.push_back(bufA.data());
        io.inputs.push_back(bufB.data());
        io.outputs.push_back(outC.data());
        io.elements = nC;
        return io;
    };
    auto rw =
        mlk::executeKernelOnBuffers(*mod, symbols, bind(outW), nullptr);
    MLK_CHECK(rw.has_value());
    if (!rw.has_value()) return;

    MLK_CHECK(backendToolchainReady());
    if (!backendToolchainReady()) return;
    mlk::BackendDriverConfig cfg;

    auto cppArt =
        mlk::buildKernelArtifact(*mod, symbols, mlk::ArtifactKind::Cpp,
                                 cfg);
    MLK_CHECK(cppArt.has_value());
    if (!cppArt.has_value()) {
        std::fprintf(stderr, "  emit/cpp: %s\n",
                     cppArt.error().message.c_str());
        return;
    }
    auto rCpp = cppArt->run(*mod, symbols, bind(outCpp));
    MLK_CHECK(rCpp.has_value());
    if (!rCpp.has_value()) {
        std::fprintf(stderr, "  run/cpp: %s\n",
                     rCpp.error().message.c_str());
        return;
    }

    auto asmArt =
        mlk::buildKernelArtifact(*mod, symbols, mlk::ArtifactKind::Asm,
                                 cfg);
    MLK_CHECK(asmArt.has_value());
    if (!asmArt.has_value()) {
        std::fprintf(stderr, "  emit/asm: %s\n",
                     asmArt.error().message.c_str());
        return;
    }
    MLK_CHECK(asmArt->isMultiDim());
    auto rAsm = asmArt->run(*mod, symbols, bind(outAsm));
    MLK_CHECK(rAsm.has_value());
    if (!rAsm.has_value()) {
        std::fprintf(stderr, "  run/asm: %s\n",
                     rAsm.error().message.c_str());
        return;
    }
    for (std::size_t i = 0; i < nC; ++i) {
        // Three-way bit-exact agreement (Rules 33/43/90).
        MLK_CHECK(outAsm[i] == outW[i]);
        MLK_CHECK(outCpp[i] == outW[i]);
    }
}

MLK_TEST(poly, asm_backend_tiled_guarded_gemm_bitexact) {
    // Tiled schedule (tile 2: full tiles + partial tails) with the
    // guarded k==0 init re-entry — the hardest codegen shape — lowered
    // to assembly and executed natively; bit-exact vs the untiled
    // transformed kernel executed by the walker.
    SymbolTable symbols;
    KernelModule base = buildGemmKernel(symbols);
    auto flatMod = buildScheduledGemm(symbols, base, 0);
    MLK_CHECK(flatMod.has_value());
    if (!flatMod.has_value()) return;
    auto tiledMod = buildScheduledGemm(symbols, base, 2);
    MLK_CHECK(tiledMod.has_value());
    if (!tiledMod.has_value()) return;

    constexpr std::size_t nA = static_cast<std::size_t>(kTestM * kTestK);
    constexpr std::size_t nB = static_cast<std::size_t>(kTestK * kTestN);
    constexpr std::size_t nC = static_cast<std::size_t>(kTestM * kTestN);
    SmallVector<double, 8> bufA(nA), bufB(nB);
    for (std::size_t i = 0; i < nA; ++i) {
        bufA[i] = static_cast<double>(i % 7) * 0.25;
    }
    for (std::size_t i = 0; i < nB; ++i) {
        bufB[i] = static_cast<double>(i % 5) * 0.5;
    }
    SmallVector<double, 8> outRef(nC, 0.0), outAsm(nC, 0.0);
    auto bind = [&](SmallVector<double, 8>& outC) {
        mlk::KernelBufferBindings io;
        io.inputs.push_back(bufA.data());
        io.inputs.push_back(bufB.data());
        io.outputs.push_back(outC.data());
        io.elements = nC;
        return io;
    };
    auto rw =
        mlk::executeKernelOnBuffers(*flatMod, symbols, bind(outRef),
                                    nullptr);
    MLK_CHECK(rw.has_value());
    if (!rw.has_value()) return;

    MLK_CHECK(backendToolchainReady());
    if (!backendToolchainReady()) return;
    mlk::BackendDriverConfig cfg;
    auto asmArt =
        mlk::buildKernelArtifact(*tiledMod, symbols,
                                 mlk::ArtifactKind::Asm, cfg);
    MLK_CHECK(asmArt.has_value());
    if (!asmArt.has_value()) {
        std::fprintf(stderr, "  emit/asm: %s\n",
                     asmArt.error().message.c_str());
        return;
    }
    auto rAsm = asmArt->run(*tiledMod, symbols, bind(outAsm));
    MLK_CHECK(rAsm.has_value());
    if (!rAsm.has_value()) {
        std::fprintf(stderr, "  run/asm: %s\n",
                     rAsm.error().message.c_str());
        return;
    }
    for (std::size_t i = 0; i < nC; ++i) {
        MLK_CHECK(outAsm[i] == outRef[i]);  // bit-exact (Rule 43)
    }
}

MLK_TEST(poly, asm_backend_guarded_accumulate_bitexact) {
    // Affine-equality guard (i == 2) + unguarded sibling, executed as an
    // assembly artifact: exactly one guarded write lands; the unguarded
    // write covers every element — the walker's per-element truth.
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer y;
    y.name = symbols.intern("y");
    y.dims = SmallVector<int64_t, 4>{6};
    y.isOutput = true;
    KernelBuffer z;
    z.name = symbols.intern("z");
    z.dims = SmallVector<int64_t, 4>{6};
    z.isOutput = true;
    const uint32_t by = km.addBuffer(y);
    const uint32_t bz = km.addBuffer(z);
    const SymbolId vi = symbols.intern("i");

    KernelNode computeG;
    computeG.op = mlk::KernelOp::Compute;
    KernelExpr five;
    five.op = mlk::MathOp::Add;
    five.a.kind = KernelOperand::Kind::Const;
    five.a.constValue = 5.0;
    computeG.exprs.push_back(five);
    KernelNode storeG;
    storeG.op = mlk::KernelOp::Store;
    storeG.bufferOut = by;
    storeG.outIndexCoeffs = SmallVector<int64_t, 4>{1};

    KernelNode computeZ;
    computeZ.op = mlk::KernelOp::Compute;
    KernelExpr two;
    two.op = mlk::MathOp::Add;
    two.a.kind = KernelOperand::Kind::Const;
    two.a.constValue = 2.0;
    computeZ.exprs.push_back(two);
    KernelNode storeZ;
    storeZ.op = mlk::KernelOp::Store;
    storeZ.bufferOut = bz;
    storeZ.outIndexCoeffs = SmallVector<int64_t, 4>{1};

    KernelNode guard;
    guard.op = mlk::KernelOp::Guard;
    guard.guardCoeffs = SmallVector<int64_t, 4>{1};
    guard.guardOffset = -2;  // i - 2 == 0

    KernelNode loop;
    loop.op = mlk::KernelOp::Loop;
    loop.var = vi;
    loop.begin = 0;
    loop.end = 6;
    {
        const uint32_t cg = km.addNode(computeG);
        const uint32_t sg = km.addNode(storeG);
        guard.children.push_back(cg);
        guard.children.push_back(sg);
        const uint32_t gid = km.addNode(guard);
        const uint32_t cz = km.addNode(computeZ);
        const uint32_t sz = km.addNode(storeZ);
        loop.children.push_back(gid);
        loop.children.push_back(cz);
        loop.children.push_back(sz);
        (void)km.addNode(loop);
    }

    MLK_CHECK(backendToolchainReady());
    if (!backendToolchainReady()) return;
    mlk::BackendDriverConfig cfg;
    auto asmArt =
        mlk::buildKernelArtifact(km, symbols, mlk::ArtifactKind::Asm, cfg);
    MLK_CHECK(asmArt.has_value());
    if (!asmArt.has_value()) {
        std::fprintf(stderr, "  emit/asm: %s\n",
                     asmArt.error().message.c_str());
        return;
    }
    SmallVector<double, 8> bufY(6, 0.0), bufZ(6, 0.0);
    mlk::KernelBufferBindings io;
    io.outputs.push_back(bufY.data());
    io.outputs.push_back(bufZ.data());
    io.elements = 6;
    auto r = asmArt->run(km, symbols, io);
    MLK_CHECK(r.has_value());
    if (!r.has_value()) {
        std::fprintf(stderr, "  run/asm: %s\n",
                     r.error().message.c_str());
        return;
    }
    for (std::size_t i = 0; i < 6; ++i) {
        MLK_CHECK(bufY[i] == (i == 2 ? 5.0 : 0.0));  // guarded write
        MLK_CHECK(bufZ[i] == 2.0);                   // unguarded write
    }
}

MLK_TEST(poly, asm_backend_reducesum_bitexact) {
    // ReduceSum synthesis class (init + accumulate nests, reduction dim
    // carried) lowered to assembly; the artifact must reproduce the
    // ascending-k accumulation order bit-exactly (Rule 90).
    SymbolTable symbols;
    mlk::DiagnosticEngine diag;
    mlk::PassContext ctx;
    ctx.symbols = &symbols;
    ctx.diag = &diag;
    ctx.tier = mlk::Tier::Tier2;
    mlk::poly::PolyWorkspace* ws = mlk::poly::createPolyWorkspace();
    ctx.polyWorkspace = ws;
    mlk::MathGraph graph(&symbols);
    mlk::passes::registerAllPasses(symbols);

    constexpr int64_t M = 32, K = 16;
    KernelModule km;
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = SmallVector<int64_t, 4>{M, K};
    a.isInput = true;
    KernelBuffer y;
    y.name = symbols.intern("Y");
    y.dims = SmallVector<int64_t, 4>{M};
    y.isOutput = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufY = km.addBuffer(y);
    KernelNode call;
    call.op = mlk::KernelOp::Call;
    call.math = mlk::MathOp::ReduceSum;
    call.bufferA = bufA;
    call.bufferOut = bufY;
    (void)km.addNode(call);
    ctx.kernelOut = &km;

    mlk::MathGraph g2(&symbols);
    auto passFn = [&](const char* name) -> mlk::Pass* {
        return mlk::PassRegistry::instance().byName(symbols.intern(name));
    };
    for (const char* name :
         {"poly.synth", "poly.scop_detect", "poly.dependence",
          "poly.schedule", "poly.tile", "poly.codegen", "poly.verify"}) {
        auto r = passFn(name)->run(ctx, g2);
        MLK_CHECK(r.has_value());
    }
    MLK_CHECK(ws->codegenValid);

    const std::size_t nA = static_cast<std::size_t>(M * K);
    SmallVector<double, 8> bufInA(nA);
    for (std::size_t i = 0; i < nA; ++i) {
        bufInA[i] = static_cast<double>((i * 11) % 17) * 0.125;
    }
    SmallVector<double, 8> outW(static_cast<std::size_t>(M), -1.0),
        outAsm(static_cast<std::size_t>(M), -1.0);
    auto bind = [&](SmallVector<double, 8>& outC) {
        mlk::KernelBufferBindings io;
        io.inputs.push_back(bufInA.data());
        io.outputs.push_back(outC.data());
        io.elements = nA;
        return io;
    };
    auto rw = mlk::executeKernelOnBuffers(km, symbols, bind(outW), nullptr);
    MLK_CHECK(rw.has_value());
    if (!rw.has_value()) {
        mlk::poly::destroyPolyWorkspace(ws);
        return;
    }

    MLK_CHECK(backendToolchainReady());
    if (!backendToolchainReady()) {
        mlk::poly::destroyPolyWorkspace(ws);
        return;
    }
    mlk::BackendDriverConfig cfg;
    auto asmArt =
        mlk::buildKernelArtifact(km, symbols, mlk::ArtifactKind::Asm, cfg);
    MLK_CHECK(asmArt.has_value());
    if (!asmArt.has_value()) {
        std::fprintf(stderr, "  emit/asm: %s\n",
                     asmArt.error().message.c_str());
        mlk::poly::destroyPolyWorkspace(ws);
        return;
    }
    auto rAsm = asmArt->run(km, symbols, bind(outAsm));
    MLK_CHECK(rAsm.has_value());
    if (!rAsm.has_value()) {
        std::fprintf(stderr, "  run/asm: %s\n",
                     rAsm.error().message.c_str());
        mlk::poly::destroyPolyWorkspace(ws);
        return;
    }
    for (std::size_t i = 0; i < outW.size(); ++i) {
        MLK_CHECK(outAsm[i] == outW[i]);  // bit-exact (Rule 43)
    }
    mlk::poly::destroyPolyWorkspace(ws);
}

MLK_TEST(poly, asm_backend_poly7_sin_bitexact) {
    // The verified "poly7" Sin family in the ASSEMBLY artifact: the
    // local helper routines must reproduce math_families.h operation
    // for operation (Cody-Waite quadrant reduction, degree-13 minimax
    // Horner residuals), so walker / C++ artifact / assembly artifact
    // agree BIT-EXACTLY including quadrant edges. Values stay inside
    // the family's verified domain [-pi, pi] (Rule 34 certificate).
    SymbolTable symbols;
    constexpr int64_t N = 64;
    KernelModule km;
    km.name = symbols.intern("poly7_sin_kernel");
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = SmallVector<int64_t, 4>{N};
    a.isInput = true;
    KernelBuffer out;
    out.name = symbols.intern("Out");
    out.dims = SmallVector<int64_t, 4>{N};
    out.isOutput = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufOut = km.addBuffer(out);
    const SymbolId vi = symbols.intern("i");

    KernelNode loop;
    loop.op = mlk::KernelOp::Loop;
    loop.var = vi;
    loop.begin = 0;
    loop.end = N;
    KernelNode compute;
    compute.op = mlk::KernelOp::Compute;
    compute.math = mlk::MathOp::Sin;
    compute.family = symbols.intern("poly7");
    KernelExpr e;
    e.op = mlk::MathOp::Sin;
    e.a.kind = KernelOperand::Kind::ElemIdx;
    e.a.index = static_cast<int64_t>(bufA);
    e.a.idxCoeffs = SmallVector<int64_t, 4>{1};
    e.a.idxOffset = 0;
    compute.exprs.push_back(e);
    KernelNode store;
    store.op = mlk::KernelOp::Store;
    store.bufferOut = bufOut;
    store.outIndexCoeffs = SmallVector<int64_t, 4>{1};
    {
        const uint32_t cid = km.addNode(compute);
        const uint32_t sid = km.addNode(store);
        loop.children.push_back(cid);
        loop.children.push_back(sid);
        (void)km.addNode(loop);
    }

    // Inputs: exact quadrant edges plus deterministic fractions across
    // [-pi, pi].
    SmallVector<double, 8> bufIn(static_cast<std::size_t>(N));
    constexpr double kPi = 3.14159265358979323846;
    const double edges[] = {0.0, kPi, -kPi, kPi / 2, -kPi / 2,
                            kPi / 4, -kPi / 4, 3 * kPi / 4, -3 * kPi / 4};
    for (int64_t i = 0; i < N; ++i) {
        const std::size_t ui = static_cast<std::size_t>(i);
        if (i < 9) {
            bufIn[ui] = edges[i];
        } else {
            const double t =
                static_cast<double>((i * 37) % 101) / 101.0;
            bufIn[ui] = -kPi + t * 2.0 * kPi;
        }
    }
    SmallVector<double, 8> outW(static_cast<std::size_t>(N), 0.0),
        outCpp(static_cast<std::size_t>(N), 0.0),
        outAsm(static_cast<std::size_t>(N), 0.0);
    auto bind = [&](SmallVector<double, 8>& outC) {
        mlk::KernelBufferBindings io;
        io.inputs.push_back(bufIn.data());
        io.outputs.push_back(outC.data());
        io.elements = N;
        return io;
    };
    auto rw = mlk::executeKernelOnBuffers(km, symbols, bind(outW), nullptr);
    MLK_CHECK(rw.has_value());
    if (!rw.has_value()) return;

    MLK_CHECK(backendToolchainReady());
    if (!backendToolchainReady()) return;
    mlk::BackendDriverConfig cfg;
    auto cppArt =
        mlk::buildKernelArtifact(km, symbols, mlk::ArtifactKind::Cpp, cfg);
    MLK_CHECK(cppArt.has_value());
    if (!cppArt.has_value()) {
        std::fprintf(stderr, "  emit/cpp: %s\n",
                     cppArt.error().message.c_str());
        return;
    }
    auto rCpp = cppArt->run(km, symbols, bind(outCpp));
    MLK_CHECK(rCpp.has_value());
    if (!rCpp.has_value()) {
        std::fprintf(stderr, "  run/cpp: %s\n",
                     rCpp.error().message.c_str());
        return;
    }
    auto asmArt =
        mlk::buildKernelArtifact(km, symbols, mlk::ArtifactKind::Asm, cfg);
    MLK_CHECK(asmArt.has_value());
    if (!asmArt.has_value()) {
        std::fprintf(stderr, "  emit/asm: %s\n",
                     asmArt.error().message.c_str());
        return;
    }
    auto rAsm = asmArt->run(km, symbols, bind(outAsm));
    MLK_CHECK(rAsm.has_value());
    if (!rAsm.has_value()) {
        std::fprintf(stderr, "  run/asm: %s\n",
                     rAsm.error().message.c_str());
        return;
    }
    for (std::size_t i = 0; i < outW.size(); ++i) {
        // Three-way bit-exact agreement (Rules 33/43/90).
        MLK_CHECK(outAsm[i] == outW[i]);
        MLK_CHECK(outCpp[i] == outW[i]);
    }
    // The family actually changed the result vs libm sin on SOME input
    // (the test would be vacuous if the dispatch were inert) — but the
    // family stays within its verified single-digit-ULP certificate.
    bool differs = false;
    for (std::size_t i = 0; i < outW.size(); ++i) {
        differs = differs || outW[i] != std::sin(bufIn[i]);
    }
    MLK_CHECK(differs);
}

#endif  // x86-64 native execution tests

MLK_TEST(poly, backend_rejects_lowered_first_nodes) {
    // Honest failure contract: Call nodes (Rule 121) and speculative
    // guards (Rule 5) are rejected by BOTH emitters — never approximated
    // silently.
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = SmallVector<int64_t, 4>{4, 4};
    a.isInput = true;
    KernelBuffer y;
    y.name = symbols.intern("Y");
    y.dims = SmallVector<int64_t, 4>{4};
    y.isOutput = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufY = km.addBuffer(y);
    KernelNode call;
    call.op = mlk::KernelOp::Call;
    call.math = mlk::MathOp::MatMul;
    call.bufferA = bufA;
    call.bufferOut = bufY;
    (void)km.addNode(call);
    auto cppR = mlk::emitCppSource(km, symbols);
    MLK_CHECK(!cppR.has_value());
    auto asmR = mlk::emitAsmSource(km, symbols);
    MLK_CHECK(!asmR.has_value());
}

MLK_TEST(poly, asm_backend_text_snapshot) {
    // Deterministic text contract of the assembly artifact (pure string
    // generation — no toolchain needed): the GEMM nest must show the
    // SSE2 scalar kernel, the affine addressing, the negative-store
    // guard, the ABI frame, and the recorded parallelism marks.
    SymbolTable symbols;
    KernelModule base = buildGemmKernel(symbols);
    auto mod = buildScheduledGemm(symbols, base, 0);
    MLK_CHECK(mod.has_value());
    if (!mod.has_value()) return;
    auto s = mlk::emitAsmSource(*mod, symbols);
    MLK_CHECK(s.has_value());
    if (!s.has_value()) {
        std::fprintf(stderr, "  emit/asm: %s\n",
                     s.error().message.c_str());
        return;
    }
    MLK_CHECK(s->find(".globl  mlk_kernel") != std::string::npos);
    MLK_CHECK(s->find(".type   mlk_kernel,@function") !=
              std::string::npos);
    MLK_CHECK(s->find("pushq   %rbp") != std::string::npos);
    MLK_CHECK(s->find("mulsd %xmm1, %xmm0") != std::string::npos);
    MLK_CHECK(s->find("addsd %xmm1, %xmm0") != std::string::npos);
    MLK_CHECK(s->find("movsd (%r10,%rax,1), %xmm0") !=
              std::string::npos);  // ElemIdx load
    MLK_CHECK(s->find("movsd %xmm0, (%r10,%rax,1)") !=
              std::string::npos);  // overwrite store
    MLK_CHECK(s->find("movsd (%r10,%rax,1), %xmm1") !=
              std::string::npos);  // accumulate load
    MLK_CHECK(s->find("testq %rax, %rax\njs .Lmlk_neg_store") !=
              std::string::npos);  // negative-flat guard
    MLK_CHECK(s->find(".Lmlk_neg_store:") != std::string::npos);
    MLK_CHECK(s->find("call ") == std::string::npos);  // pure arithmetic
    MLK_CHECK(s->find(".section .note.GNU-stack") != std::string::npos);
    // The schedule's parallel outer rows are RECORDED (Rule 148).
    MLK_CHECK(s->find("# Rule 148 recorded: parallel loops=") !=
              std::string::npos);
    // Determinism: re-emission is byte-identical (Rule 53).
    auto s2 = mlk::emitAsmSource(*mod, symbols);
    MLK_CHECK(s2.has_value() && *s2 == *s);
}

MLK_TEST_MAIN("poly")
