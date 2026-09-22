// Polyhedral engine unit tests (Rules 10, 33, 42, 90): exact rational
// arithmetic, affine expressions, Presburger set operations, FM emptiness,
// lexmin/lexmax witness verification, affine map image/preimage.
#include <optional>

#include "mlk/poly/poly.h"
#include "mlk/runtime/execution.h"
#include "mlk/backend/cpp_emitter.h"
#include "mlk/backend/asm_emitter.h"
#include "mlk/backend/cuda_emitter.h"
#include "mlk/backend/backend_driver.h"

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

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

MLK_TEST(poly, lexmax_coupled_prefix_substitution) {
    // Regression: 0 <= j <= i <= 7. Fixing i=7 must SUBSTITUTE i out of the
    // coupled row (i - j >= 0) before the univariate interval for j is
    // read; the old code read that row as (-j >= 0) and returned (7,0).
    const VarSpace sp = dims2();
    Polyhedron p;
    p.space = sp;
    SmallVector<int64_t, 8> ri0;
    ri0.push_back(1);
    ri0.push_back(0);
    p.addInequality(std::move(ri0), 0);  // i >= 0
    SmallVector<int64_t, 8> ri7;
    ri7.push_back(-1);
    ri7.push_back(0);
    p.addInequality(std::move(ri7), 7);  // i <= 7
    SmallVector<int64_t, 8> rj0;
    rj0.push_back(0);
    rj0.push_back(1);
    p.addInequality(std::move(rj0), 0);  // j >= 0
    SmallVector<int64_t, 8> rji;
    rji.push_back(1);
    rji.push_back(-1);
    p.addInequality(std::move(rji), 0);  // i - j >= 0  (j <= i)
    PresburgerSet s;
    s.space = sp;
    s.disjuncts.push_back(std::move(p));
    auto mx = s.lexMax();
    MLK_CHECK(mx.has_value());
    MLK_CHECK(mx->has_value());
    if (mx->has_value()) {
        MLK_CHECK_EQ((**mx)[0], 7);
        MLK_CHECK_EQ((**mx)[1], 7);  // was 0 on the unsubstituted prefix
        MLK_CHECK(s.containsPoint(**mx));
    }
    auto mn = s.lexMin();
    MLK_CHECK(mn.has_value());
    MLK_CHECK(mn->has_value());
    if (mn->has_value()) {
        MLK_CHECK_EQ((**mn)[0], 0);
        MLK_CHECK_EQ((**mn)[1], 0);
    }
}

MLK_TEST(poly, lexmin_one_sided_unbounded_dim) {
    // Regression: {0 <= i <= 3, j >= i - 9} has no upper bound on j. lexMin
    // only needs the search-direction (lower) bound: it must return the
    // witness (0, -9). The old code errored on ANY unbounded interval and
    // integerLeFormFeasible reported that as "no solution". lexMax on this
    // set has no maximum (unbounded search direction) -> honest error.
    const VarSpace sp = dims2();
    Polyhedron p;
    p.space = sp;
    SmallVector<int64_t, 8> ri0;
    ri0.push_back(1);
    ri0.push_back(0);
    p.addInequality(std::move(ri0), 0);  // i >= 0
    SmallVector<int64_t, 8> ri3;
    ri3.push_back(-1);
    ri3.push_back(0);
    p.addInequality(std::move(ri3), 3);  // i <= 3
    SmallVector<int64_t, 8> rjge;
    rjge.push_back(-1);
    rjge.push_back(1);
    p.addInequality(std::move(rjge), 9);  // -i + j - 9 >= 0  (j >= i - 9)
    PresburgerSet s;
    s.space = sp;
    s.disjuncts.push_back(std::move(p));
    auto mn = s.lexMin();
    MLK_CHECK(mn.has_value());
    MLK_CHECK(mn->has_value());
    if (mn->has_value()) {
        MLK_CHECK_EQ((**mn)[0], 0);
        MLK_CHECK_EQ((**mn)[1], -9);
        MLK_CHECK(s.containsPoint(**mn));  // witness against the original
    }
    auto mx = s.lexMax();
    MLK_CHECK(!mx.has_value());  // error: no maximum exists (Rule 67)
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
    store1.accum = mlk::AccumMode::Add;

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
    MLK_CHECK(s1.accum == mlk::AccumMode::Add);

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
    store.accum = mlk::AccumMode::Add;
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
    MLK_CHECK(initS.accum == mlk::AccumMode::None);
    MLK_CHECK_EQ(accC0.op, mlk::KernelOp::Compute);
    MLK_CHECK_EQ(accS0.accum, mlk::AccumMode::Add);
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
    MLK_CHECK_EQ(accS.accum, mlk::AccumMode::Add);
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
            mlk::PassRegistry::instance().byName(symbols, symbols.intern(name));
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
        return mlk::PassRegistry::instance().byName(symbols, symbols.intern(name));
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
        return mlk::PassRegistry::instance().byName(symbols, symbols.intern(name));
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

// ---------------------------------------------------------------------------
// createDirs (driver): the recursive mkdir -p behind the workdir base,
// the artifact dir, and the fast-kernel cache root. A fresh checkout or
// a fresh --cache-dir has NO pre-existing parent chain — the previous
// single-level mkdir turned that perfectly normal first-run state into
// a spurious IoError for every nested path (the cache tests only passed
// on machines carrying build leftovers, breaking fresh CI runs).
// ---------------------------------------------------------------------------

MLK_TEST(poly, create_dirs_recursive_semantics) {
    // Run-unique root under the cwd (ctest's build tree — never /tmp,
    // matching the driver's own workdir policy).
    const std::string base =
        std::string("./fk_createDirs_test-") +
        std::to_string(static_cast<long long>(
            ::std::chrono::steady_clock::now().time_since_epoch().count()));
    const std::string deep = base + "/a/b/c";
    const auto isDir = [](const std::string& p) {
        struct ::stat st {};
        return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
    };

    // 1. A missing multi-level chain is created whole.
    auto made = mlk::createDirs(deep);
    MLK_CHECK(made.has_value());
    MLK_CHECK(isDir(deep));
    MLK_CHECK(isDir(base + "/a/b"));

    // 2. Idempotent: an existing chain is success, not an error.
    auto again = mlk::createDirs(deep);
    MLK_CHECK(again.has_value());

    // 3. An empty path is an honest InvalidArgument, never a silent ok.
    auto empty = mlk::createDirs("");
    MLK_CHECK(!empty.has_value());

    // 4. A regular file blocking a chain component is an ERROR — EEXIST
    // on a non-directory must not pass for success.
    const std::string blocked = base + "/f";
    FILE* blocker = ::std::fopen(blocked.c_str(), "wb");
    MLK_CHECK(blocker != nullptr);
    if (blocker != nullptr) {
        ::std::fclose(blocker);
        auto throughFile = mlk::createDirs(blocked + "/sub");
        MLK_CHECK(!throughFile.has_value());
    }

    // 5. Trailing-slash form terminates at the same directory.
    auto slash = mlk::createDirs(base + "/t/");
    MLK_CHECK(slash.has_value());
    MLK_CHECK(isDir(base + "/t"));

    // Best-effort cleanup: the test leaves nothing behind on success.
    ::rmdir((base + "/t").c_str());
    ::remove(blocked.c_str());
    ::rmdir((base + "/a/b/c").c_str());
    ::rmdir((base + "/a/b").c_str());
    ::rmdir((base + "/a").c_str());
    ::rmdir(base.c_str());
}

/// Legacy 1-D elementwise module: out[i] = A[i] * B[i] over a dynamic
/// bound (the historical ABI form). Shared by the legacy artifact test
/// and the CUDA multi-axis text test (which asserts the legacy form
/// does NOT gain the multi-dim geometry helper).
[[nodiscard]] KernelModule buildLegacyMulModule(SymbolTable& symbols) {
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
    return km;
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
    KernelModule km = buildLegacyMulModule(symbols);

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
        return mlk::PassRegistry::instance().byName(symbols, symbols.intern(name));
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

#if defined(__x86_64__) || defined(_M_X64)
MLK_TEST(poly, asm_backend_packed_text_and_structure) {
    // Packed-2 emission contract (pure text — no toolchain): the
    // scheduled GEMM's innermost output-column loops qualify (the
    // reduction rides an OUTER level; the innermost store is stride-1 in
    // its own var), so the artifact must contain the pair forms, the
    // packed control labels, the remainder loop, and the header record.
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
    // Packed pair ops + pair addressing are present.
    MLK_CHECK(s->find("mulpd %xmm1, %xmm0") != std::string::npos);
    MLK_CHECK(s->find("addpd %xmm0, %xmm1") != std::string::npos);
    MLK_CHECK(s->find("movupd (%r10,%rax,1), %xmm0") !=
              std::string::npos);  // contiguous pair load
    MLK_CHECK(s->find("movupd %xmm1, (%r10,%rax,1)") !=
              std::string::npos);  // pair accumulate store
    MLK_CHECK(s->find("unpcklpd %xmm0, %xmm0") !=
              std::string::npos);  // shared-operand broadcast
    // Control structure: main loop, its end, the scalar remainder.
    // (Label numbers share one counter with the scalar loop labels, so
    // only the prefix is asserted.)
    MLK_CHECK(s->find(".Lpkmain") != std::string::npos);
    MLK_CHECK(s->find("addq $2, ") != std::string::npos);
    MLK_CHECK(s->find(".Lpktail") != std::string::npos);
    MLK_CHECK(s->find(".Lpkend") != std::string::npos);
    // The scalar forms survive in the remainder loop and the
    // non-packable nests (init statements, outer reduction bands).
    MLK_CHECK(s->find("mulsd %xmm1, %xmm0") != std::string::npos);
    MLK_CHECK(s->find("addsd %xmm1, %xmm0") != std::string::npos);
    // Header records the packed count (Rule 148-style reporting).
    MLK_CHECK(s->find("# Packed-2 execution: ") != std::string::npos);
    // No libm calls anywhere in this pure-arithmetic kernel.
    MLK_CHECK(s->find("call ") == std::string::npos);
    // Determinism: re-emission is byte-identical (Rule 53).
    auto s2 = mlk::emitAsmSource(*mod, symbols);
    MLK_CHECK(s2.has_value() && *s2 == *s);
}

MLK_TEST(poly, asm_backend_packed_remainder_bitexact) {
    // Odd innermost trip (N=5): the packed main loop covers the even
    // prefix, the scalar remainder loop runs the final iteration with
    // the ORIGINAL body — the artifact must agree with the walker
    // bit-exact (the remainder path is where an off-by-one would show).
    SymbolTable symbols;
    KernelModule base = buildGemmKernel(symbols);
    auto mod = buildScheduledGemm(symbols, base, 0);
    MLK_CHECK(mod.has_value());
    if (!mod.has_value()) return;
    auto s = mlk::emitAsmSource(*mod, symbols);
    MLK_CHECK(s.has_value());
    if (!s.has_value()) return;
    // N=5 -> some packed loop must carry a non-empty remainder.
    MLK_CHECK(s->find(".Lpktail") != std::string::npos);

    constexpr std::size_t nA = static_cast<std::size_t>(kTestM * kTestK);
    constexpr std::size_t nB = static_cast<std::size_t>(kTestK * kTestN);
    constexpr std::size_t nC = static_cast<std::size_t>(kTestM * kTestN);
    SmallVector<double, 8> bufA(nA), bufB(nB);
    for (std::size_t i = 0; i < nA; ++i) {
        bufA[i] = static_cast<double>(i % 7) * 0.25 - 0.5;
    }
    for (std::size_t i = 0; i < nB; ++i) {
        bufB[i] = static_cast<double>(i % 5) * 0.5;
    }
    SmallVector<double, 8> outW(nC, 0.0), outAsm(nC, 0.0);
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
    auto asmArt = mlk::buildKernelArtifact(*mod, symbols,
                                           mlk::ArtifactKind::Asm, cfg);
    MLK_CHECK(asmArt.has_value());
    if (!asmArt.has_value()) {
        std::fprintf(stderr, "  emit/asm: %s\n",
                     asmArt.error().message.c_str());
        return;
    }
    auto rAsm = asmArt->run(*mod, symbols, bind(outAsm));
    MLK_CHECK(rAsm.has_value());
    if (!rAsm.has_value()) {
        std::fprintf(stderr, "  run/asm: %s\n",
                     rAsm.error().message.c_str());
        return;
    }
    for (std::size_t i = 0; i < nC; ++i) {
        MLK_CHECK(outAsm[i] == outW[i]);  // Rule 43: bit-exact
    }
}

MLK_TEST(poly, asm_backend_packed_fallback_div_stays_scalar) {
    // Graceful fallback (Rule 30): a Div chain in the innermost body has
    // no pair form (the zero-divisor select), so the loop must keep the
    // scalar form — no packed labels, no pair ops — and stay bit-exact.
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = SmallVector<int64_t, 4>{3, 8};
    a.isInput = true;
    KernelBuffer b;
    b.name = symbols.intern("B");
    b.dims = SmallVector<int64_t, 4>{3, 8};
    b.isInput = true;
    KernelBuffer y;
    y.name = symbols.intern("Y");
    y.dims = SmallVector<int64_t, 4>{3, 8};
    y.isOutput = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufB = km.addBuffer(b);
    const uint32_t bufY = km.addBuffer(y);
    const SymbolId vi = symbols.intern("i");
    const SymbolId vj = symbols.intern("j");
    // innermost j: Y[i][j] = A[i][j] / B[i][j] (b != 0 walker select).
    KernelNode compute;
    compute.op = mlk::KernelOp::Compute;
    KernelExpr div;
    div.op = mlk::MathOp::Div;
    div.a.kind = KernelOperand::Kind::ElemIdx;
    div.a.index = static_cast<int64_t>(bufA);
    div.a.idxCoeffs = SmallVector<int64_t, 4>{8, 1};
    div.b.kind = KernelOperand::Kind::ElemIdx;
    div.b.index = static_cast<int64_t>(bufB);
    div.b.idxCoeffs = SmallVector<int64_t, 4>{8, 1};
    compute.exprs.push_back(div);
    KernelNode store;
    store.op = mlk::KernelOp::Store;
    store.bufferOut = bufY;
    store.outIndexCoeffs = SmallVector<int64_t, 4>{8, 1};
    KernelNode jLoop;
    jLoop.op = mlk::KernelOp::Loop;
    jLoop.var = vj;
    jLoop.begin = 0;
    jLoop.end = 8;
    {
        const uint32_t c = km.addNode(compute);
        const uint32_t st = km.addNode(store);
        jLoop.children.push_back(c);
        jLoop.children.push_back(st);
    }
    KernelNode iLoop;
    iLoop.op = mlk::KernelOp::Loop;
    iLoop.var = vi;
    iLoop.begin = 0;
    iLoop.end = 3;
    {
        const uint32_t j = km.addNode(jLoop);
        iLoop.children.push_back(j);
    }
    (void)km.addNode(iLoop);

    auto s = mlk::emitAsmSource(km, symbols);
    MLK_CHECK(s.has_value());
    if (!s.has_value()) {
        std::fprintf(stderr, "  emit/asm: %s\n",
                     s.error().message.c_str());
        return;
    }
    // The Div gate rejects the loop: scalar form only.
    MLK_CHECK(s->find(".Lpkmain") == std::string::npos);
    MLK_CHECK(s->find("movupd") == std::string::npos);
    MLK_CHECK(s->find("divsd %xmm1, %xmm0") != std::string::npos);
    MLK_CHECK(s->find("# Packed-2 execution") == std::string::npos);

    constexpr std::size_t n = static_cast<std::size_t>(3 * 8);
    SmallVector<double, 8> bufInA(n), bufInB(n);
    for (std::size_t i = 0; i < n; ++i) {
        bufInA[i] = static_cast<double>(i) * 0.5 - 2.0;
        bufInB[i] = static_cast<double>((i % 5)) * 0.25 + 0.5;  // nonzero
    }
    SmallVector<double, 8> outW(n, 0.0), outAsm(n, 0.0);
    auto bind = [&](SmallVector<double, 8>& outC) {
        mlk::KernelBufferBindings io;
        io.inputs.push_back(bufInA.data());
        io.inputs.push_back(bufInB.data());
        io.outputs.push_back(outC.data());
        io.elements = n;
        return io;
    };
    auto rw = mlk::executeKernelOnBuffers(km, symbols, bind(outW), nullptr);
    MLK_CHECK(rw.has_value());
    if (!rw.has_value()) return;

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
    auto rAsm = asmArt->run(km, symbols, bind(outAsm));
    MLK_CHECK(rAsm.has_value());
    if (!rAsm.has_value()) {
        std::fprintf(stderr, "  run/asm: %s\n",
                     rAsm.error().message.c_str());
        return;
    }
    for (std::size_t i = 0; i < n; ++i) {
        MLK_CHECK(outAsm[i] == outW[i]);  // Rule 43: bit-exact
    }
}

#endif  // __x86_64__

MLK_TEST(poly, softmax_synth_pipeline_bitexact) {
    // Softmax synthesis milestone: baseline Call(Softmax) -> poly.synth
    // (stable-form chain with materialized rowmax/exp/sum temps, four
    // sibling k-bands) -> SCoP with the chain-final-read dependences ->
    // the scheduler finds NO legal fused schedule (reading a completed
    // reduction chain requires band separation; the band-shift roadmap
    // item) -> the kernel keeps its synthesized form -> bit-exact vs
    // BOTH the reference kernel and a hand-computed stable softmax
    // (Rule 90: identical ops in identical order).
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

    constexpr int64_t M = 24, K = 16;
    KernelModule km;
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = SmallVector<int64_t, 4>{M, K};
    a.isInput = true;
    KernelBuffer y;
    y.name = symbols.intern("Y");
    y.dims = SmallVector<int64_t, 4>{M, K};
    y.isOutput = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufY = km.addBuffer(y);
    KernelNode call;
    call.op = mlk::KernelOp::Call;
    call.math = mlk::MathOp::Softmax;
    call.bufferA = bufA;
    call.bufferOut = bufY;
    (void)km.addNode(call);
    ctx.kernelOut = &km;

    const std::size_t n = static_cast<std::size_t>(M * K);
    SmallVector<double, 8> bufIn(n), baseY(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        bufIn[i] = static_cast<double>((i * 13) % 19) * 0.25 - 2.0;
    }
    // Baseline execution first (Call path: rowwise max/exp/sum/div in
    // ascending-k order).
    {
        mlk::KernelBufferBindings io;
        io.inputs.push_back(bufIn.data());
        io.outputs.push_back(baseY.data());
        io.elements = n;
        auto br = mlk::executeKernelOnBuffers(km, symbols, io, nullptr);
        MLK_CHECK(br.has_value());
        if (!br.has_value()) {
            mlk::poly::destroyPolyWorkspace(ws);
            return;
        }
    }

    mlk::MathGraph g2(&symbols);
    auto passFn = [&](const char* name) -> mlk::Pass* {
        return mlk::PassRegistry::instance().byName(symbols, symbols.intern(name));
    };
    for (const char* name :
         {"poly.synth", "poly.scop_detect", "poly.dependence",
          "poly.schedule", "poly.tile", "poly.codegen", "poly.verify"}) {
        auto r = passFn(name)->run(ctx, g2);
        MLK_CHECK(r.has_value());
        if (!r.has_value()) {
            mlk::poly::destroyPolyWorkspace(ws);
            return;
        }
    }
    // Synthesis structure: three materialized temps + the six-statement
    // band nest (16 nodes: 6 compute + 6 store + 4 k-loops + 1 m-loop).
    MLK_CHECK_EQ(km.buffers.size(), std::size_t{5});
    MLK_CHECK_EQ(km.nodes.size(), std::size_t{17});
    int tempCount = 0;
    for (const auto& b : km.buffers) {
        if (b.isTemp) ++tempCount;
    }
    MLK_CHECK_EQ(tempCount, 3);
    // Honest schedule boundary: the chain-final-read dependences force
    // band separation (roadmap); no fused schedule is claimed.
    MLK_CHECK(!ws->scheduleValid);
    MLK_CHECK(!ws->codegenValid);
    MLK_CHECK_EQ(ws->scop.statements.size(), std::size_t{6});

    // Transformed (synthesized, unfused) execution: bit-exact vs BOTH
    // the baseline call and the hand-computed reference.
    SmallVector<double, 8> bufOutY(n, -1.0);
    mlk::KernelBufferBindings io;
    io.inputs.push_back(bufIn.data());
    io.outputs.push_back(bufOutY.data());
    io.elements = n;
    auto r = mlk::executeKernelOnBuffers(km, symbols, io, nullptr);
    MLK_CHECK(r.has_value());
    if (!r.has_value()) {
        mlk::poly::destroyPolyWorkspace(ws);
        return;
    }
    int mismatches = 0;
    for (int64_t i = 0; i < M && mismatches == 0; ++i) {
        double mx = -INFINITY;
        for (int64_t k = 0; k < K; ++k) {
            const double v = bufIn[static_cast<std::size_t>(i * K + k)];
            mx = (v > mx) ? v : mx;
        }
        double s = 0.0;
        SmallVector<double, 8> es(static_cast<std::size_t>(K));
        for (int64_t k = 0; k < K; ++k) {
            es[static_cast<std::size_t>(k)] =
                std::exp(bufIn[static_cast<std::size_t>(i * K + k)] - mx);
            s += es[static_cast<std::size_t>(k)];
        }
        for (int64_t k = 0; k < K; ++k) {
            const double ref = s != 0.0
                                   ? es[static_cast<std::size_t>(k)] / s
                                   : 0.0;
            const double got =
                bufOutY[static_cast<std::size_t>(i * K + k)];
            const double base =
                baseY[static_cast<std::size_t>(i * K + k)];
            if (got != ref || base != ref) ++mismatches;  // Rule 43
        }
    }
    MLK_CHECK_EQ(mismatches, 0);
    mlk::poly::destroyPolyWorkspace(ws);
}

MLK_TEST(poly, softmax_shapes_k1_edge_bitexact) {
    // Shape sweep of the synthesized class, including the K == 1 edge
    // (rowmax == x, e == exp(0) == 1, s == 1, out == 1 exactly) and a
    // tall row (K == 7 over M == 3).
    SymbolTable symbols;
    struct Dim {
        int64_t m, k;
    };
    const Dim dims[] = {{5, 1}, {3, 7}, {8, 4}};
    for (const Dim& d : dims) {
        KernelModule km;
        KernelBuffer a;
        a.name = symbols.intern("A");
        a.dims = SmallVector<int64_t, 4>{d.m, d.k};
        a.isInput = true;
        KernelBuffer y;
        y.name = symbols.intern("Y");
        y.dims = SmallVector<int64_t, 4>{d.m, d.k};
        y.isOutput = true;
        const uint32_t bufA = km.addBuffer(a);
        const uint32_t bufY = km.addBuffer(y);
        KernelNode call;
        call.op = mlk::KernelOp::Call;
        call.math = mlk::MathOp::Softmax;
        call.bufferA = bufA;
        call.bufferOut = bufY;
        (void)km.addNode(call);

        const std::size_t n = static_cast<std::size_t>(d.m * d.k);
        SmallVector<double, 8> in(n), base(n, 0.0), out(n, -1.0);
        for (std::size_t i = 0; i < n; ++i) {
            in[i] = static_cast<double>((i * 7) % 11) * 0.5 - 2.5;
        }
        mlk::KernelBufferBindings baseIo;
        baseIo.inputs.push_back(in.data());
        baseIo.outputs.push_back(base.data());
        baseIo.elements = static_cast<int64_t>(n);
        auto br = mlk::executeKernelOnBuffers(km, symbols, baseIo, nullptr);
        MLK_CHECK(br.has_value());

        // Synth only (the pipeline outcome is covered by the flagship
        // test; here the synthesized nest itself must be exact for every
        // shape).
        mlk::PassContext ctx;
        ctx.symbols = &symbols;
        ctx.tier = mlk::Tier::Tier2;
        mlk::poly::PolyWorkspace* ws = mlk::poly::createPolyWorkspace();
        ctx.polyWorkspace = ws;
        ctx.kernelOut = &km;
        mlk::MathGraph g(&symbols);
        mlk::passes::registerAllPasses(symbols);
        auto* synth = mlk::PassRegistry::instance().byName(symbols, symbols.intern("poly.synth"));
        auto r = synth->run(ctx, g);
        MLK_CHECK(r.has_value() && r->changed);
        if (r.has_value() && r->changed) {
            mlk::KernelBufferBindings io;
            io.inputs.push_back(in.data());
            io.outputs.push_back(out.data());
            io.elements = static_cast<int64_t>(n);
            auto er = mlk::executeKernelOnBuffers(km, symbols, io, nullptr);
            MLK_CHECK(er.has_value());
            if (er.has_value()) {
                int mismatches = 0;
                for (std::size_t i = 0; i < n && mismatches == 0; ++i) {
                    if (out[i] != base[i]) ++mismatches;  // Rule 43
                }
                MLK_CHECK_EQ(mismatches, 0);
            }
        }
        mlk::poly::destroyPolyWorkspace(ws);
    }
}

MLK_TEST(poly, max_accumulate_store_semantics) {
    // The Max-accumulate store is the rowmax primitive: the running
    // value survives unless the incoming value compares GREATER. NaN
    // never replaces it ((NaN > cur) is false); +/-0 ties keep the
    // current slot; -inf init loses to the first finite value. The
    // hand-built module below exercises each case per row.
    SymbolTable symbols;
    KernelModule km;
    // in[m][0]: row 0 = NaN (never replaces the +inf... see below),
    // row 1 = -0.0 tie, row 2 = normal max.
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = SmallVector<int64_t, 4>{3, 1};
    a.isInput = true;
    KernelBuffer o;
    o.name = symbols.intern("O");
    o.dims = SmallVector<int64_t, 4>{3, 1};
    o.isOutput = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufO = km.addBuffer(o);
    // m { O[m] = -inf; k { O[m] max= A[m,k] } }
    KernelNode c0;
    c0.op = mlk::KernelOp::Compute;
    {
        KernelExpr e;
        e.op = mlk::MathOp::Add;
        e.a.kind = KernelOperand::Kind::Const;
        e.a.constValue = -INFINITY;
        e.b.kind = KernelOperand::Kind::Const;
        e.b.constValue = 0.0;
        c0.exprs.push_back(e);
    }
    KernelNode s0;
    s0.op = mlk::KernelOp::Store;
    s0.bufferOut = bufO;
    s0.outIndexCoeffs = SmallVector<int64_t, 4>{1};
    KernelNode c1;
    c1.op = mlk::KernelOp::Compute;
    {
        KernelExpr e;
        e.op = mlk::MathOp::Sub;
        e.a.kind = KernelOperand::Kind::ElemIdx;
        e.a.index = static_cast<int64_t>(bufA);
        e.a.idxCoeffs = SmallVector<int64_t, 4>{1, 0};  // A[m,k]
        e.b.kind = KernelOperand::Kind::Const;
        e.b.constValue = 0.0;
        c1.exprs.push_back(e);
    }
    KernelNode s1;
    s1.op = mlk::KernelOp::Store;
    s1.bufferOut = bufO;
    s1.outIndexCoeffs = SmallVector<int64_t, 4>{1, 0};
    s1.accum = mlk::AccumMode::Max;
    KernelNode kl;
    kl.op = mlk::KernelOp::Loop;
    kl.var = symbols.intern("k");
    kl.begin = 0;
    kl.end = 1;
    kl.children.push_back(2);
    kl.children.push_back(3);
    KernelNode ml;
    ml.op = mlk::KernelOp::Loop;
    ml.var = symbols.intern("m");
    ml.begin = 0;
    ml.end = 3;
    ml.children.push_back(0);
    ml.children.push_back(1);
    ml.children.push_back(4);
    KernelNode nodes[6] = {c0, s0, c1, s1, kl, ml};
    for (const auto& nd : nodes) (void)km.addNode(nd);

    SmallVector<double, 8> in{NAN, -0.0, 2.5};
    SmallVector<double, 8> out(3, 0.0);
    mlk::KernelBufferBindings io;
    io.inputs.push_back(in.data());
    io.outputs.push_back(out.data());
    io.elements = 3;
    auto r = mlk::executeKernelOnBuffers(km, symbols, io, nullptr);
    MLK_CHECK(r.has_value());
    if (!r.has_value()) return;
    // Exact select semantics (documented in kernel_ir.h):
    //   row 0: (NaN > -inf) is false -> -inf survives (NaN never
    //          replaces the running value);
    //   row 1: (-0.0 > -inf) is true -> the slot becomes -0.0 (and the
    //          +/-0 tie rule would keep an existing ±0 on equal compare);
    //   row 2: plain maximum.
    MLK_CHECK(std::isnan(in[0]));
    MLK_CHECK(out[0] == -INFINITY);  // NaN never replaces
    MLK_CHECK(out[1] == -0.0);       // -0.0 > -inf replaces exactly
    MLK_CHECK(out[2] == 2.5);
}

MLK_TEST(poly, softmax_native_three_way_bitexact) {
    // Native temp ABI milestone: the synthesized softmax module (three
    // materialized temps, Max-accumulate rowmax stores) now runs
    // THREE-WAY — buffer walker vs C++ artifact vs x86-64 asm artifact
    // — with bit-exact outputs including the adversarial NaN and +/-0
    // rows (Rule 90: identical ops in identical order; Rule 43: exact
    // float equality semantics, bit patterns compared so -0.0 and NaN
    // payloads count).
    SymbolTable symbols;
    mlk::DiagnosticEngine diag;
    mlk::PassContext ctx;
    ctx.symbols = &symbols;
    ctx.diag = &diag;
    ctx.tier = mlk::Tier::Tier2;
    mlk::poly::PolyWorkspace* ws = mlk::poly::createPolyWorkspace();
    ctx.polyWorkspace = ws;
    mlk::passes::registerAllPasses(symbols);

    constexpr int64_t M = 12, K = 8;
    KernelModule km;
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = SmallVector<int64_t, 4>{M, K};
    a.isInput = true;
    KernelBuffer y;
    y.name = symbols.intern("Y");
    y.dims = SmallVector<int64_t, 4>{M, K};
    y.isOutput = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufY = km.addBuffer(y);
    KernelNode call;
    call.op = mlk::KernelOp::Call;
    call.math = mlk::MathOp::Softmax;
    call.bufferA = bufA;
    call.bufferOut = bufY;
    (void)km.addNode(call);
    ctx.kernelOut = &km;

    const std::size_t n = static_cast<std::size_t>(M * K);
    SmallVector<double, 8> bufIn(n), walkerY(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        bufIn[i] = static_cast<double>((i * 13) % 19) * 0.25 - 2.0;
    }
    // Adversarial rows (walker rowmax select is order-insensitive but
    // the engines must still agree bit-for-bit):
    // row 9: a NaN input never enters the rowmax ((NaN > cur) is
    // false), poisons exp/sum/div — every engine propagates NaN to the
    // whole row.
    bufIn[static_cast<std::size_t>(9 * K + 2)] = std::nan("");
    // row 10: -0.0 replaces -inf, then +0.0 ties and the CURRENT slot
    // (-0.0) is kept — the sign bit of the rowmax is observable through
    // the (v > cur) select only, so outputs must match bit-exactly.
    for (int64_t k = 0; k < K; ++k) {
        bufIn[static_cast<std::size_t>(10 * K + static_cast<std::size_t>(
                                                    k))] =
            (k % 2 == 0) ? -0.0 : 0.0;
    }
    // row 11: the tie-after-negative form (rowmax becomes -0.0 and a
    // later +0.0 must NOT replace it).
    bufIn[static_cast<std::size_t>(11 * K)] = -1.0;
    bufIn[static_cast<std::size_t>(11 * K + 1)] = -0.0;
    bufIn[static_cast<std::size_t>(11 * K + 2)] = 0.0;
    bufIn[static_cast<std::size_t>(11 * K + 3)] = -2.0;

    // Baseline execution (Call path) — the oracle the artifacts must
    // match bit-for-bit after synthesis.
    {
        mlk::KernelBufferBindings io;
        io.inputs.push_back(bufIn.data());
        io.outputs.push_back(walkerY.data());
        io.elements = n;
        auto br = mlk::executeKernelOnBuffers(km, symbols, io, nullptr);
        MLK_CHECK(br.has_value());
        if (!br.has_value()) {
            mlk::poly::destroyPolyWorkspace(ws);
            return;
        }
    }

    // Synthesize: baseline Call -> materialized-temp statement chain.
    mlk::MathGraph g2(&symbols);
    auto* synth =
        mlk::PassRegistry::instance().byName(symbols,
                                             symbols.intern("poly.synth"));
    auto sr = synth->run(ctx, g2);
    MLK_CHECK(sr.has_value() && sr->changed);
    if (!sr.has_value()) {
        std::fprintf(stderr, "  synth error: %s\n",
                     sr.error().message.c_str());
    } else if (!sr->changed) {
        std::fprintf(stderr, "  synth: no change\n");
    }
    if (!sr.has_value() || !sr->changed) {
        mlk::poly::destroyPolyWorkspace(ws);
        return;
    }
    MLK_CHECK_EQ(km.buffers.size(), std::size_t{5});

    // Walker over the SYNTHESIZED form: the artifact contract to match.
    SmallVector<double, 8> synthY(n, 0.0);
    {
        mlk::KernelBufferBindings io;
        io.inputs.push_back(bufIn.data());
        io.outputs.push_back(synthY.data());
        io.elements = n;
        auto r = mlk::executeKernelOnBuffers(km, symbols, io, nullptr);
        MLK_CHECK(r.has_value());
        if (!r.has_value()) {
            mlk::poly::destroyPolyWorkspace(ws);
            return;
        }
    }
    // Synthesized walker == baseline Call (the differential guarantee).
    MLK_CHECK(std::memcmp(synthY.data(), walkerY.data(),
                          n * sizeof(double)) == 0);

    // Native three-way: both artifacts run through the driver, which
    // materializes the temp scratch exactly like the walker.
    MLK_CHECK(backendToolchainReady());
    if (!backendToolchainReady()) {
        mlk::poly::destroyPolyWorkspace(ws);
        return;
    }
    mlk::BackendDriverConfig cfg;
    for (const mlk::ArtifactKind kind : {mlk::ArtifactKind::Cpp,
                                         mlk::ArtifactKind::Asm}) {
        auto loaded = mlk::buildKernelArtifact(km, symbols, kind, cfg);
        MLK_CHECK(loaded.has_value());
        if (!loaded.has_value()) {
            std::fprintf(stderr, "  emit/kind=%d: %s\n",
                         static_cast<int>(kind),
                         loaded.error().message.c_str());
            mlk::poly::destroyPolyWorkspace(ws);
            return;
        }
        SmallVector<double, 8> nativeY(n, 0.0);
        mlk::KernelBufferBindings io;
        io.inputs.push_back(bufIn.data());
        io.outputs.push_back(nativeY.data());
        io.elements = n;
        auto r = loaded->run(km, symbols, io);
        MLK_CHECK(r.has_value());
        if (!r.has_value()) {
            std::fprintf(stderr, "  run/kind=%d: %s\n",
                         static_cast<int>(kind),
                         r.error().message.c_str());
            mlk::poly::destroyPolyWorkspace(ws);
            return;
        }
        // Bit-exact vs the walker (NaN payloads and -0.0 sign bits
        // included — memcmp, never ==).
        MLK_CHECK(std::memcmp(nativeY.data(), walkerY.data(),
                              n * sizeof(double)) == 0);
    }
    mlk::poly::destroyPolyWorkspace(ws);
}

MLK_TEST(poly, max_accumulate_store_native_bitexact) {
    // The Max select is directly observable through a temp: S1
    // accumulates rowmax into temp T (AccumMode::Max), S2 copies T to Y
    // through the exact passthrough (Sub(x, 0.0) preserves -0.0 and NaN
    // payloads). The zero-init ABI means the +0.0 slot is the RUNNING
    // value an adversarial row must beat — pinning the walker select
    // ((v > cur) ? v : cur) in BOTH artifacts: NaN never replaces,
    // +/-0 ties keep the current slot's sign bit (std::max/vmaxsd
    // would flip exactly here).
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = SmallVector<int64_t, 4>{4, 4};
    a.isInput = true;
    KernelBuffer yy;
    yy.name = symbols.intern("Y");
    yy.dims = SmallVector<int64_t, 4>{4};
    yy.isOutput = true;
    KernelBuffer t;
    t.name = symbols.intern("T");
    t.dims = SmallVector<int64_t, 4>{4};
    t.isTemp = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufY = km.addBuffer(yy);
    const uint32_t bufT = km.addBuffer(t);
    const SymbolId vm = symbols.intern("m");
    const SymbolId vk = symbols.intern("k");

    KernelNode loopM;
    loopM.op = mlk::KernelOp::Loop;
    loopM.var = vm;
    loopM.begin = 0;
    loopM.end = 4;
    KernelNode loopK;
    loopK.op = mlk::KernelOp::Loop;
    loopK.var = vk;
    loopK.begin = 0;
    loopK.end = 4;
    // S0 (init): T[m] = -inf — Add(-inf, 0.0) is the exact constant;
    // without it the zero-init slot (+0.0) could never go negative and
    // the +/-0 tie would be unobservable (max only increases).
    KernelNode initCompute;
    initCompute.op = mlk::KernelOp::Compute;
    initCompute.exprs.push_back(KernelExpr{});
    initCompute.exprs[0].op = mlk::MathOp::Add;
    initCompute.exprs[0].a.kind = KernelOperand::Kind::Const;
    initCompute.exprs[0].a.constValue = -INFINITY;
    initCompute.exprs[0].b.kind = KernelOperand::Kind::Const;
    initCompute.exprs[0].b.constValue = 0.0;
    KernelNode initStore;
    initStore.op = mlk::KernelOp::Store;
    initStore.bufferOut = bufT;
    initStore.outIndexCoeffs = SmallVector<int64_t, 4>{1};
    KernelNode maxCompute;
    maxCompute.op = mlk::KernelOp::Compute;
    maxCompute.exprs.push_back(KernelExpr{});
    maxCompute.exprs[0].op = mlk::MathOp::Sub;
    maxCompute.exprs[0].a.kind = KernelOperand::Kind::ElemIdx;
    maxCompute.exprs[0].a.index = static_cast<int64_t>(bufA);
    maxCompute.exprs[0].a.idxCoeffs = SmallVector<int64_t, 4>{4, 1};
    maxCompute.exprs[0].b.kind = KernelOperand::Kind::Const;
    maxCompute.exprs[0].b.constValue = 0.0;  // exact identity
    KernelNode maxStore;
    maxStore.op = mlk::KernelOp::Store;
    maxStore.bufferOut = bufT;
    maxStore.outIndexCoeffs = SmallVector<int64_t, 4>{1, 0};
    maxStore.accum = mlk::AccumMode::Max;
    KernelNode copyCompute;
    copyCompute.op = mlk::KernelOp::Compute;
    copyCompute.exprs.push_back(KernelExpr{});
    copyCompute.exprs[0].op = mlk::MathOp::Sub;
    copyCompute.exprs[0].a.kind = KernelOperand::Kind::ElemIdx;
    copyCompute.exprs[0].a.index = static_cast<int64_t>(bufT);
    copyCompute.exprs[0].a.idxCoeffs = SmallVector<int64_t, 4>{1};
    copyCompute.exprs[0].b.kind = KernelOperand::Kind::Const;
    copyCompute.exprs[0].b.constValue = 0.0;  // exact identity
    KernelNode copyStore;
    copyStore.op = mlk::KernelOp::Store;
    copyStore.bufferOut = bufY;
    copyStore.outIndexCoeffs = SmallVector<int64_t, 4>{1};
    {
        const uint32_t ic = km.addNode(initCompute);
        const uint32_t is = km.addNode(initStore);
        const uint32_t lc = km.addNode(maxCompute);
        const uint32_t ls = km.addNode(maxStore);
        loopK.children.push_back(lc);
        loopK.children.push_back(ls);
        const uint32_t lk = km.addNode(loopK);
        const uint32_t cc = km.addNode(copyCompute);
        const uint32_t cs = km.addNode(copyStore);
        loopM.children.push_back(ic);
        loopM.children.push_back(is);
        loopM.children.push_back(lk);
        loopM.children.push_back(cc);
        loopM.children.push_back(cs);
        (void)km.addNode(loopM);
    }

    // Inputs (explicit -inf init band, mirroring the softmax synth's
    // S0; the walker + both artifacts must agree bit-for-bit):
    // row 0: 1.0 then 2.0 win; NaN and -0.0 never replace  -> 2.0
    // row 1: all-NaN: the -inf init survives               -> -inf
    // row 2: -1.0 replaces, then the +/-0 pair: -0.0
    //        replaces -1.0, +0.0 does NOT replace -0.0     -> -0.0
    // row 3: 1.0 wins                                      -> 1.0
    SmallVector<double, 8> in{
        1.0, std::nan(""), -0.0, 2.0,          // row 0
        std::nan(""), std::nan(""), std::nan(""),
        std::nan(""),                          // row 1
        -1.0, -0.0, 0.0, -2.0,                 // row 2
        0.5, 0.25, 1.0, -8.0                   // row 3
    };
    SmallVector<double, 8> walkerY(4, -99.0);
    {
        mlk::KernelBufferBindings io;
        io.inputs.push_back(in.data());
        io.outputs.push_back(walkerY.data());
        io.elements = 4;
        auto r = mlk::executeKernelOnBuffers(km, symbols, io, nullptr);
        MLK_CHECK(r.has_value());
        if (!r.has_value()) return;
    }
    // Walker semantics (also documents the contract for the artifacts):
    const double expected[4] = {2.0, -INFINITY, -0.0, 1.0};
    for (int64_t m = 0; m < 4; ++m) {
        MLK_CHECK(std::memcmp(&walkerY[static_cast<std::size_t>(m)],
                              &expected[m], sizeof(double)) == 0);
    }

    // Native three-way.
    MLK_CHECK(backendToolchainReady());
    if (!backendToolchainReady()) return;
    mlk::BackendDriverConfig cfg;
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
        SmallVector<double, 8> nativeY(4, -99.0);
        mlk::KernelBufferBindings io;
        io.inputs.push_back(in.data());
        io.outputs.push_back(nativeY.data());
        io.elements = 4;
        auto r = loaded->run(km, symbols, io);
        MLK_CHECK(r.has_value());
        if (!r.has_value()) {
            std::fprintf(stderr, "  run/kind=%d: %s\n",
                         static_cast<int>(kind),
                         r.error().message.c_str());
            return;
        }
        // memcmp: -0.0 vs +0.0 (and NaN payloads) MUST count.
        MLK_CHECK(std::memcmp(nativeY.data(), walkerY.data(),
                              4 * sizeof(double)) == 0);
    }
}

// ---------------------------------------------------------------------------
// Round 19: the CUDA backend (spec #GPU-backend; ADR-0008). The .cu text
// artifact is fully machine-checkable WITHOUT a GPU: structure asserts pin
// the grid mapping (proven-parallel collapse), the ABI wrapper (device
// memory behind the SAME mlk_kernel ABI), the --fmad=false build contract,
// and the declared exactness policy. Compile/run paths are gated by the
// recorded nvcc probe — honest skip, never a silent green.
// ---------------------------------------------------------------------------

/// Cold, cached probe: live CUDA tests require nvcc on PATH. On this
/// platform the tests SKIP honestly (the skip condition itself is the
/// recorded check); every OTHER failure fails the test.
bool cudaToolchainReady() {
    static const bool ready = [] {
        mlk::GpuBackendDriverConfig cfg;
        return mlk::cudaToolchainAvailable(cfg);
    }();
    return ready;
}

MLK_TEST(poly, cuda_emission_gemm_structure) {
    // Scheduled GEMM -> .cu text: one __global__ per root, the collapsed
    // [i,j] parallel chain (proven by the scheduler) becomes a flat grid
    // with row-major decomposition, the carried k runs serially inside
    // the thread, and the host wrapper manages device memory behind the
    // SAME ABI. No OpenMP pragmas exist on the device path; no std::
    // calls either (device code uses the unqualified math functions).
    SymbolTable symbols;
    KernelModule base = buildGemmKernel(symbols);
    auto mod = buildScheduledGemm(symbols, base, 0);
    MLK_CHECK(mod.has_value());
    if (!mod.has_value()) return;
    auto src = mlk::emitCudaSource(*mod, symbols);
    MLK_CHECK(src.has_value());
    if (!src.has_value()) {
        std::fprintf(stderr, "  emit: %s\n", src.error().message.c_str());
        return;
    }
    const std::string& s = *src;
    // Device kernel + host wrapper + the exactness/flags contract.
    MLK_CHECK(s.find("__global__ void mlk_dev_0(") != std::string::npos);
    MLK_CHECK(s.find("extern \"C\" int mlk_kernel(") != std::string::npos);
    MLK_CHECK(s.find("--fmad=false") != std::string::npos);
    // Grid collapse: flat thread id + bounds guard + row-major
    // decomposition. The scheduled GEMM collapses its outermost
    // parallel chain (the split k segments stop deeper collapse), so
    // the decomposition is a direct assignment; deeper chains divide
    // by the suffix product (mlk_rem %= covers that form).
    MLK_CHECK(s.find("blockIdx.x") != std::string::npos);
    MLK_CHECK(s.find("threadIdx.x") != std::string::npos);
    MLK_CHECK(s.find("if (mlk_flat >= mlk_total) return;") !=
              std::string::npos);
    MLK_CHECK(s.find("= mlk_rem;") != std::string::npos ||
              s.find("mlk_rem %=") != std::string::npos);
    // Device memory management behind the ABI (the wrapper owns it).
    MLK_CHECK(s.find("cudaMalloc") != std::string::npos);
    MLK_CHECK(s.find("cudaMemcpy") != std::string::npos);
    MLK_CHECK(s.find("cudaMemset") != std::string::npos);
    MLK_CHECK(s.find("cudaFree") != std::string::npos);
    MLK_CHECK(s.find("cudaDeviceSynchronize") != std::string::npos);
    // The negative-flat structural check (ABI code 1 <-> walker
    // InvalidGraph) at the store site.
    MLK_CHECK(s.find("if (mlk_flat < 0) { *mlk_status = 1; return; }") !=
              std::string::npos);
    // Device-exact policy recorded for a pure mul/add GEMM.
    MLK_CHECK(s.find("Exactness policy: bit-exact") != std::string::npos);
    MLK_CHECK(s.find("#pragma omp") == std::string::npos);
    MLK_CHECK(s.find("std::") == std::string::npos);
    // Rule 148 recorded line exists.
    MLK_CHECK(s.find("Rule 148 recorded") != std::string::npos);
}

MLK_TEST(poly, cuda_emission_deterministic) {
    // Byte-identical re-emission (the asm snapshot contract): the .cu
    // text is a pure function of the module (hash + recorded marks; no
    // clocks, no addresses).
    SymbolTable symbols;
    KernelModule base = buildGemmKernel(symbols);
    auto mod = buildScheduledGemm(symbols, base, 0);
    MLK_CHECK(mod.has_value());
    if (!mod.has_value()) return;
    auto a = mlk::emitCudaSource(*mod, symbols);
    auto b = mlk::emitCudaSource(*mod, symbols);
    MLK_CHECK(a.has_value() && b.has_value());
    if (!a.has_value() || !b.has_value()) return;
    MLK_CHECK(*a == *b);
}

MLK_TEST(poly, cuda_emission_legacy_elementwise) {
    // Legacy 1-D module: one thread per element over the caller's n,
    // the static-range guard mirrors the 1-D executor's loop bounds,
    // and the wrapper sizes device buffers by n (the shared dynamic
    // bound of the legacy ABI).
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
    loop.end = mlk::constants::kKernelLoopDynamicBound;
    KernelNode compute;
    compute.op = mlk::KernelOp::Compute;
    compute.math = mlk::MathOp::Mul;
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
    auto src = mlk::emitCudaSource(km, symbols);
    MLK_CHECK(src.has_value());
    if (!src.has_value()) {
        std::fprintf(stderr, "  emit: %s\n", src.error().message.c_str());
        return;
    }
    const std::string& s = *src;
    MLK_CHECK(s.find("__global__ void mlk_dev_0(") != std::string::npos);
    MLK_CHECK(s.find("extern \"C\" void mlk_kernel(") != std::string::npos);
    MLK_CHECK(s.find("blockIdx.x * (int64_t)blockDim.x") !=
              std::string::npos);
    MLK_CHECK(s.find("mlk_i[") == std::string::npos);  // no affine machinery
    // Policy: a pure Mul module is device-exact.
    MLK_CHECK(mlk::cudaArtifactBitExactPolicy(km, symbols));
}

MLK_TEST(poly, cuda_emission_multiaxis_text) {
    // Round 20 multi-axis geometry: the emitted artifact carries (a)
    // the generated host geometry helper (deterministic greedy, never
    // clamps, 1-D flat fallback), (b) per-root padded-trip arrays
    // feeding it, (c) dim3 launches over the helper's axis products,
    // and (d) the FULL hardware flat linearization (all six axes) in
    // the device text — invariant to the assignment by radix
    // associativity. The legacy 1-D path keeps its own n-based flat
    // grid and must NOT gain the helper.
    SymbolTable symbols;
    KernelModule base = buildGemmKernel(symbols);
    auto mod = buildScheduledGemm(symbols, base, 0);
    MLK_CHECK(mod.has_value());
    if (!mod.has_value()) return;
    auto src = mlk::emitCudaSource(*mod, symbols);
    MLK_CHECK(src.has_value());
    if (!src.has_value()) {
        std::fprintf(stderr, "  emit: %s\n", src.error().message.c_str());
        return;
    }
    const std::string& s = *src;
    // The geometry helper exists exactly once, before the wrapper.
    const std::size_t helper = s.find("static void mlk_assign_geometry(");
    MLK_CHECK(helper != std::string::npos);
    MLK_CHECK(s.find("static void mlk_assign_geometry(",
                     helper + 1) == std::string::npos);
    MLK_CHECK(s.find("extern \"C\" int mlk_kernel(") != std::string::npos);
    MLK_CHECK(s.find("extern \"C\" int mlk_kernel(") > helper);
    // Declared caps in the generated greedy (the envelope is text, so
    // it is pinned): block 1024/1024/64 cumulative 1024, grid
    // 2^31-1/65535/65535, division-form cap checks.
    MLK_CHECK(s.find("(mlk_ax == 2) ? 64 : 1024") != std::string::npos);
    MLK_CHECK(s.find("mlk_bprod <= 1024 / mlk_t[mlk_i]") !=
              std::string::npos);
    MLK_CHECK(s.find("2147483647LL") != std::string::npos);
    MLK_CHECK(s.find("65535LL") != std::string::npos);
    MLK_CHECK(s.find("(mlk_total + 1023) / 1024") != std::string::npos);
    // Per-root trips array + the geometry call + dim3 launch shapes.
    // The untiled scheduled GEMM's collapse chain is ONE level [i]:
    // the piecewise-split k segments are SIBLING subtrees, so j is
    // duplicated per segment and the interior-single-loop-child rule
    // stops the collapse at i (the split is the recorded structural
    // form of the guarded re-entry).
    MLK_CHECK(s.find("const int64_t mlk_trips[] = {") != std::string::npos);
    MLK_CHECK(s.find("mlk_assign_geometry(mlk_trips, 1, mlk_g, mlk_b)") !=
              std::string::npos);
    MLK_CHECK(s.find("const dim3 mlk_grid((unsigned)mlk_g[0], "
                     "(unsigned)mlk_g[1], (unsigned)mlk_g[2]);") !=
              std::string::npos);
    MLK_CHECK(s.find("const dim3 mlk_block((unsigned)mlk_b[0], "
                     "(unsigned)mlk_b[1], (unsigned)mlk_b[2]);") !=
              std::string::npos);
    // The device flat id is the FULL hardware linearization: every
    // index space except gridDim.z appears (gridDim.z is ABSENT BY
    // MATH — bz is the slowest axis and has no suffix radix in the
    // linearization; gridDim.y/blockIdx.z prove the multi-axis form).
    MLK_CHECK(s.find("(int64_t)gridDim.y") != std::string::npos);
    MLK_CHECK(s.find("(int64_t)blockIdx.z") != std::string::npos);
    MLK_CHECK(s.find("(int64_t)blockDim.y") != std::string::npos);
    MLK_CHECK(s.find("(int64_t)blockDim.z") != std::string::npos);
    MLK_CHECK(s.find("(int64_t)blockIdx.y") != std::string::npos);
    MLK_CHECK(s.find("(int64_t)threadIdx.y") != std::string::npos);
    MLK_CHECK(s.find("(int64_t)threadIdx.z") != std::string::npos);
    // The guard + row-major decomposition survive unchanged.
    MLK_CHECK(s.find("if (mlk_flat >= mlk_total) return;") !=
              std::string::npos);
    MLK_CHECK(s.find("= mlk_rem;") != std::string::npos ||
              s.find("mlk_rem %=") != std::string::npos);
    // Byte-identical re-emission with the helper in the text.
    auto again = mlk::emitCudaSource(*mod, symbols);
    MLK_CHECK(again.has_value() && *again == s);

    // The legacy 1-D form has no geometry helper (its flat grid comes
    // from n directly; the multi-axis machinery is multi-dim only).
    SymbolTable legacySymbols;
    KernelModule legacy = buildLegacyMulModule(legacySymbols);
    auto legacySrc = mlk::emitCudaSource(legacy, legacySymbols);
    MLK_CHECK(legacySrc.has_value());
    if (legacySrc.has_value()) {
        MLK_CHECK(legacySrc->find("mlk_assign_geometry") ==
                  std::string::npos);
        MLK_CHECK(legacySrc->find("blockIdx.x * (int64_t)blockDim.x") !=
                  std::string::npos);
    }
}

MLK_TEST(poly, cuda_emission_multiaxis_tiled_gemm_text) {
    // Tiled scheduled GEMM: the collapsed chain is the tile/point
    // prefix [ti, i] — the split k segments are sibling subtrees and
    // stop deeper collapse, so the trips array carries TWO padded
    // trips: the tile level's is exact (2 for tile 2 over M = 4) and
    // the POINT level's is the interval max over the tile box (4 = the
    // full M — partial tiles pad). Both feed the geometry helper.
    SymbolTable symbols;
    KernelModule base = buildGemmKernel(symbols);
    auto mod = buildScheduledGemm(symbols, base, 2);
    MLK_CHECK(mod.has_value());
    if (!mod.has_value()) return;
    auto src = mlk::emitCudaSource(*mod, symbols);
    MLK_CHECK(src.has_value());
    if (!src.has_value()) {
        std::fprintf(stderr, "  emit: %s\n", src.error().message.c_str());
        return;
    }
    const std::string& s = *src;
    // The tiled chain is the tile/point prefix [ti, i] (the split k
    // segments stop deeper collapse): the tile level's padded trip is
    // exact (2 for tile 2 over M = 4) and the POINT level's padded
    // trip is the interval max over the tile box (4 = the full M —
    // partial tiles pad). Both feed the geometry helper.
    MLK_CHECK(s.find("mlk_assign_geometry(mlk_trips, 2, mlk_g, mlk_b)") !=
              std::string::npos);
    MLK_CHECK(s.find("mlk_tmax0 = ") != std::string::npos);
    MLK_CHECK(s.find("mlk_tmax1 = ") != std::string::npos);
    // Byte-identical re-emission (tiled form, helper included).
    auto again = mlk::emitCudaSource(*mod, symbols);
    MLK_CHECK(again.has_value() && *again == s);
}

// ---------------------------------------------------------------------------
// Round 21: shared-memory slab reuse (docs/polyhedral_spec.md
// #GPU-backend). The plan (slab_plan.h) qualifies module-wide
// read-only buffers whose read-form hulls are dims-only; the C++
// mirror artifact behaviorally pins the value logic bit-exact against
// the walker; the CUDA twin is structure-verified (no nvcc here —
// declared boundary).
// ---------------------------------------------------------------------------

/// The scheduled GEMM's roots + module-wide store table, shared by the
/// round-21 slab tests.
void slabTestSetup(const mlk::KernelModule& mod,
                   std::vector<uint32_t>& roots,
                   std::vector<bool>& stored) {
    roots.clear();
    stored.assign(mod.buffers.size(), false);
    for (const KernelNode& n : mod.nodes) {
        if (n.op == mlk::KernelOp::Store &&
            n.bufferOut < mod.buffers.size()) {
            stored[n.bufferOut] = true;
        }
    }
    for (uint32_t i = 0; i < mod.nodes.size(); ++i) {
        bool referenced = false;
        for (const KernelNode& n : mod.nodes) {
            for (const uint32_t c : n.children) {
                referenced = referenced || c == i;
            }
        }
        if (!referenced) roots.push_back(i);
    }
}

MLK_TEST(poly, slab_plan_gemm_analysis) {
    // Planner contract on the scheduled GEMM (tile 2): the read-only
    // proof qualifies A and B (the accumulate target C is refused and
    // RECORDED), the constant-folded hulls are exact, and the plan is
    // a pure function of the module.
    SymbolTable symbols;
    KernelModule base = buildGemmKernel(symbols);
    auto mod = buildScheduledGemm(symbols, base, 2);
    MLK_CHECK(mod.has_value());
    if (!mod.has_value()) return;
    std::vector<uint32_t> roots;
    std::vector<bool> stored;
    slabTestSetup(*mod, roots, stored);
    MLK_CHECK_EQ(roots.size(), static_cast<std::size_t>(1));
    std::vector<std::string> dimsName;
    for (uint32_t bid = 0; bid < mod->buffers.size(); ++bid) {
        dimsName.push_back("buf" + std::to_string(bid) + "_dims");
    }
    mlk::SlabEmitOptions opts;
    opts.sharedMemSlabs = true;
    auto plans = mlk::planRootSlabs(*mod, roots, dimsName, stored, opts);
    MLK_CHECK(plans.has_value());
    if (!plans.has_value()) return;
    MLK_CHECK_EQ(plans->size(), static_cast<std::size_t>(1));
    const auto& plan = (*plans)[0];
    MLK_CHECK(plan.notes.empty());
    MLK_CHECK_EQ(plan.slabs.size(), static_cast<std::size_t>(2));
    if (plan.slabs.size() != 2) return;
    // Buffer-id order: A (id 0) then B (id 1); C never qualifies.
    MLK_CHECK_EQ(plan.slabs[0].buffer, static_cast<uint32_t>(0));
    MLK_CHECK_EQ(plan.slabs[1].buffer, static_cast<uint32_t>(1));
    // Constant-folded hulls over the padded tile box: the reads cover
    // exactly the full A and B rectangles [0, M*K) / [0, K*N) — the
    // split-k segments' per-symbol hulls fold into one exact span.
    MLK_CHECK(plan.slabs[0].minText == "0");
    MLK_CHECK(plan.slabs[0].spanText == "12");
    MLK_CHECK(plan.slabs[1].minText == "0");
    MLK_CHECK(plan.slabs[1].spanText == "15");
    // The elems texts are the dims products (the load's bounds guard).
    MLK_CHECK(plan.slabs[0].elemsText ==
              "((buf0_dims[0]) * buf0_dims[1])");
    MLK_CHECK(plan.slabs[1].elemsText ==
              "((buf1_dims[0]) * buf1_dims[1])");
    // Pure function of the module: re-planning is identical.
    auto again = mlk::planRootSlabs(*mod, roots, dimsName, stored, opts);
    MLK_CHECK(again.has_value() && again->size() == 1 &&
              (*again)[0].slabs.size() == 2);
    if (again.has_value() && again->size() == 1 &&
        (*again)[0].slabs.size() == 2) {
        for (std::size_t k = 0; k < 2; ++k) {
            MLK_CHECK((*again)[0].slabs[k].minText ==
                      plan.slabs[k].minText);
            MLK_CHECK((*again)[0].slabs[k].spanText ==
                      plan.slabs[k].spanText);
        }
    }
    // Options off: empty plans, no analysis.
    auto off = mlk::planRootSlabs(*mod, roots, dimsName, stored,
                                  mlk::SlabEmitOptions{});
    MLK_CHECK(off.has_value() && off->size() == 1 &&
              (*off)[0].slabs.empty());
}

MLK_TEST(poly, cuda_emission_smem_structure) {
    // The CUDA slab twin: dynamic shared decl, live-flag hole
    // discipline (no early return before the barrier), cooperative
    // load with the in-bounds guard, the folded span, the redirected
    // reads, the wrapper's runtime pick, and the header recording.
    // The default options stay byte-identical to the historical form.
    SymbolTable symbols;
    KernelModule base = buildGemmKernel(symbols);
    auto mod = buildScheduledGemm(symbols, base, 2);
    MLK_CHECK(mod.has_value());
    if (!mod.has_value()) return;
    mlk::SlabEmitOptions opts;
    opts.sharedMemSlabs = true;
    auto src = mlk::emitCudaSource(*mod, symbols, opts);
    MLK_CHECK(src.has_value());
    if (!src.has_value()) {
        std::fprintf(stderr, "  emit: %s\n", src.error().message.c_str());
        return;
    }
    const std::string& s = *src;
    // Twin kernels: the plain form AND the slab twin.
    MLK_CHECK(s.find("__global__ void mlk_dev_0(") != std::string::npos);
    MLK_CHECK(s.find("__global__ void mlk_dev_0_sm(") !=
              std::string::npos);
    // Slab prologue: one dynamic shared array.
    MLK_CHECK(s.find("extern __shared__ double mlk_smem[];") !=
              std::string::npos);
    // The slab twin replaces the flat guard with the live flag...
    MLK_CHECK(s.find("const bool mlk_live = mlk_flat < mlk_total;") !=
              std::string::npos);
    MLK_CHECK(s.find("mlk_live = mlk_live && ((") != std::string::npos);
    // ...the plain twin keeps it.
    MLK_CHECK(s.find("if (mlk_flat >= mlk_total) return;") !=
              std::string::npos);
    // Cooperative load: block-rank stride over the folded span with
    // the in-bounds guard; B's slab sits at A's span offset.
    MLK_CHECK(s.find("mlk_u < 12; mlk_u += mlk_bl") != std::string::npos);
    MLK_CHECK(s.find("mlk_idx >= 0 && mlk_idx < (((A_dims[0]) * "
                     "A_dims[1])))") != std::string::npos);
    MLK_CHECK(s.find("mlk_smem[(12) + mlk_u] = B[mlk_idx];") !=
              std::string::npos);
    MLK_CHECK(s.find("__syncthreads();") != std::string::npos);
    // Redirected reads (position = form - hull min, offset folded in).
    MLK_CHECK(s.find("mlk_smem[((3*") != std::string::npos);
    MLK_CHECK(s.find("mlk_smem[(12) + ((5*") != std::string::npos);
    // The wrapper's runtime pick: slab twin within the declared
    // budget, plain form otherwise.
    MLK_CHECK(s.find("const int64_t mlk_smem_bytes = (int64_t)sizeof("
                     "double) * ((12) + (15));") != std::string::npos);
    MLK_CHECK(s.find("mlk_smem_bytes >= 0 && mlk_smem_bytes <= 49152") !=
              std::string::npos);
    MLK_CHECK(s.find("mlk_dev_0_sm<<<mlk_grid, mlk_block, "
                     "mlk_smem_bytes>>>") != std::string::npos);
    // Header recording (Rule 148).
    MLK_CHECK(s.find("// Slab emission (round 21): 1 root(s) with slab "
                     "twins") != std::string::npos);
    // The body under the live flag: the store-flat early return stays
    // (all barriers already passed — no divergence).
    MLK_CHECK(s.find("if (mlk_flat < 0) { *mlk_status = 1; return; }") !=
              std::string::npos);
    // Default options: byte-identical to the historical emission and
    // free of any slab scaffolding.
    auto plainA = mlk::emitCudaSource(*mod, symbols);
    auto plainB = mlk::emitCudaSource(*mod, symbols,
                                      mlk::SlabEmitOptions{});
    MLK_CHECK(plainA.has_value() && plainB.has_value() &&
              *plainA == *plainB);
    MLK_CHECK(plainA->find("__syncthreads") == std::string::npos);
    MLK_CHECK(plainA->find("_sm(") == std::string::npos);
}

MLK_TEST(poly, cuda_emission_smem_deterministic) {
    // Byte-identical re-emission with the slab option on (both
    // emitters): the plan and the scaffolding names are pure
    // functions of the module.
    SymbolTable symbols;
    KernelModule base = buildGemmKernel(symbols);
    auto mod = buildScheduledGemm(symbols, base, 2);
    MLK_CHECK(mod.has_value());
    if (!mod.has_value()) return;
    mlk::SlabEmitOptions opts;
    opts.sharedMemSlabs = true;
    auto a = mlk::emitCudaSource(*mod, symbols, opts);
    auto b = mlk::emitCudaSource(*mod, symbols, opts);
    MLK_CHECK(a.has_value() && b.has_value() && *a == *b);
    auto c = mlk::emitCppSource(*mod, symbols, opts);
    auto d = mlk::emitCppSource(*mod, symbols, opts);
    MLK_CHECK(c.has_value() && d.has_value() && *c == *d);
}

MLK_TEST(poly, slab_cpp_mirror_gemm_bitexact) {
    // THE behavioral anchor for the slab value logic: the C++ mirror
    // artifact (preload into scratch + redirected reads, RAII guard)
    // must be bit-exact vs the walker on the tiled GEMM — partial
    // tiles, the padded hulls, and the split-k segments included. The
    // CUDA twin shares the plan and the redirect verbatim, so this
    // pins the value logic on hardware the tree cannot reach (no nvcc
    // here); the device prologue is structure-verified separately.
    SymbolTable symbols;
    KernelModule base = buildGemmKernel(symbols);
    for (const int64_t tile : {int64_t{0}, int64_t{2}}) {
        auto mod = buildScheduledGemm(symbols, base, tile);
        MLK_CHECK(mod.has_value());
        if (!mod.has_value()) return;
        constexpr std::size_t nA =
            static_cast<std::size_t>(kTestM * kTestK);
        constexpr std::size_t nB =
            static_cast<std::size_t>(kTestK * kTestN);
        constexpr std::size_t nC =
            static_cast<std::size_t>(kTestM * kTestN);
        SmallVector<double, 8> bufA(nA), bufB(nB);
        for (std::size_t i = 0; i < nA; ++i) {
            bufA[i] = static_cast<double>(i % 7) * 0.25;
        }
        for (std::size_t i = 0; i < nB; ++i) {
            bufB[i] = static_cast<double>(i % 5) * 0.5;
        }
        SmallVector<double, 8> outW(nC, 0.0), outCpp(nC, 0.0);
        auto bind = [&](SmallVector<double, 8>& outC) {
            mlk::KernelBufferBindings io;
            io.inputs.push_back(bufA.data());
            io.inputs.push_back(bufB.data());
            io.outputs.push_back(outC.data());
            io.elements = nC;
            return io;
        };
        auto rw = mlk::executeKernelOnBuffers(*mod, symbols, bind(outW),
                                              nullptr);
        MLK_CHECK(rw.has_value());
        if (!rw.has_value()) return;
        // The mirror text carries the scaffolding (recorded + named).
        mlk::SlabEmitOptions opts;
        opts.sharedMemSlabs = true;
        auto src = mlk::emitCppSource(*mod, symbols, opts);
        MLK_CHECK(src.has_value());
        if (!src.has_value()) return;
        MLK_CHECK(src->find("Slab emission (round 21)") !=
                  std::string::npos);
        MLK_CHECK(src->find("mlk_SlabGuard") != std::string::npos);
        MLK_CHECK(src->find("#include <cstdlib>") != std::string::npos);
        MLK_CHECK_EQ(backendToolchainReady(), true);
        if (!backendToolchainReady()) return;
        mlk::BackendDriverConfig cfg;
        cfg.slabs = opts;
        auto loaded = mlk::buildKernelArtifact(*mod, symbols,
                                               mlk::ArtifactKind::Cpp,
                                               cfg);
        MLK_CHECK(loaded.has_value());
        if (!loaded.has_value()) {
            std::fprintf(stderr, "  emit/tile=%lld: %s\n",
                         static_cast<long long>(tile),
                         loaded.error().message.c_str());
            return;
        }
        auto r = loaded->run(*mod, symbols, bind(outCpp));
        MLK_CHECK(r.has_value());
        if (!r.has_value()) {
            std::fprintf(stderr, "  run/tile=%lld: %s\n",
                         static_cast<long long>(tile),
                         r.error().message.c_str());
            return;
        }
        for (std::size_t i = 0; i < nC; ++i) {
            MLK_CHECK(outCpp[i] == outW[i]);
        }
    }
}

MLK_TEST(poly, slab_plan_stored_buffer_refused) {
    // A module where the read buffer is ALSO a store target: the
    // read-only proof fails, the buffer keeps its global reads, the
    // skip is RECORDED (Rule 148), and the CUDA emission with slabs
    // enabled has no twin for that root (it degrades to the plain
    // form, honestly).
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = SmallVector<int64_t, 4>{4, 4};
    a.isInput = true;
    KernelBuffer out;
    out.name = symbols.intern("Out");
    out.dims = SmallVector<int64_t, 4>{4, 4};
    out.isOutput = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufOut = km.addBuffer(out);
    (void)bufOut;  // written only through the store node's target id
    const SymbolId vi = symbols.intern("i");
    const SymbolId vj = symbols.intern("j");
    KernelNode compute;
    compute.op = mlk::KernelOp::Compute;
    KernelExpr ea;
    ea.op = mlk::MathOp::Add;
    ea.a.kind = mlk::KernelOperand::Kind::ElemIdx;
    ea.a.index = static_cast<int64_t>(bufA);
    ea.a.idxCoeffs = SmallVector<int64_t, 4>{4, 1};  // 4*i + j
    compute.exprs.push_back(ea);
    KernelNode store;
    store.op = mlk::KernelOp::Store;
    store.bufferOut = bufA;  // A is BOTH read and written
    store.outIndexCoeffs = SmallVector<int64_t, 4>{4, 1};
    {
        const uint32_t cid = km.addNode(compute);
        const uint32_t sid = km.addNode(store);
        KernelNode jLoop;
        jLoop.op = mlk::KernelOp::Loop;
        jLoop.var = vj;
        jLoop.begin = 0;
        jLoop.end = 4;
        jLoop.children.push_back(cid);
        jLoop.children.push_back(sid);
        const uint32_t jid = km.addNode(jLoop);
        KernelNode iLoop;
        iLoop.op = mlk::KernelOp::Loop;
        iLoop.var = vi;
        iLoop.begin = 0;
        iLoop.end = 4;
        iLoop.parallel = true;
        iLoop.children.push_back(jid);
        (void)km.addNode(iLoop);
    }
    std::vector<uint32_t> roots;
    std::vector<bool> stored;
    slabTestSetup(km, roots, stored);
    MLK_CHECK_EQ(roots.size(), static_cast<std::size_t>(1));
    std::vector<std::string> dimsName;
    for (uint32_t bid = 0; bid < km.buffers.size(); ++bid) {
        dimsName.push_back("buf" + std::to_string(bid) + "_dims");
    }
    mlk::SlabEmitOptions opts;
    opts.sharedMemSlabs = true;
    auto plans = mlk::planRootSlabs(km, roots, dimsName, stored, opts);
    MLK_CHECK(plans.has_value());
    if (!plans.has_value()) return;
    MLK_CHECK_EQ(plans->size(), static_cast<std::size_t>(1));
    MLK_CHECK((*plans)[0].slabs.empty());
    MLK_CHECK_EQ((*plans)[0].notes.size(), static_cast<std::size_t>(1));
    if (!(*plans)[0].notes.empty()) {
        MLK_CHECK((*plans)[0].notes[0].find("read-only proof failed") !=
                  std::string::npos);
    }
    // CUDA emission: no slab twin anywhere; the note is recorded in
    // the generated header.
    auto src = mlk::emitCudaSource(km, symbols, opts);
    MLK_CHECK(src.has_value());
    if (!src.has_value()) return;
    MLK_CHECK(src->find("_sm(") == std::string::npos);
    MLK_CHECK(src->find("read-only proof failed") != std::string::npos);
    MLK_CHECK(src->find("// Slab emission (round 21): 0 root(s)") !=
              std::string::npos);
}

MLK_TEST(poly, cuda_emission_smem_serial_root) {
    // A serial (non-parallel-marked) root with a read-only buffer:
    // the slab twin exists for the serial form too — same cooperative
    // load (degenerate 1x1 geometry), same barrier discipline, and
    // the wrapper's runtime pick. Serial kernels keep the plain
    // one-thread body (no live flag needed — no padding holes).
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = SmallVector<int64_t, 4>{2, 2};
    a.isInput = true;
    KernelBuffer out;
    out.name = symbols.intern("Out");
    out.dims = SmallVector<int64_t, 4>{2, 2};
    out.isOutput = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufOut = km.addBuffer(out);
    const SymbolId vi = symbols.intern("i");
    const SymbolId vj = symbols.intern("j");
    KernelNode compute;
    compute.op = mlk::KernelOp::Compute;
    KernelExpr ea;
    ea.op = mlk::MathOp::Add;
    ea.a.kind = mlk::KernelOperand::Kind::ElemIdx;
    ea.a.index = static_cast<int64_t>(bufA);
    ea.a.idxCoeffs = SmallVector<int64_t, 4>{2, 1};  // 2*i + j
    compute.exprs.push_back(ea);
    KernelNode store;
    store.op = mlk::KernelOp::Store;
    store.bufferOut = bufOut;
    store.outIndexCoeffs = SmallVector<int64_t, 4>{2, 1};
    {
        const uint32_t cid = km.addNode(compute);
        const uint32_t sid = km.addNode(store);
        KernelNode jLoop;
        jLoop.op = mlk::KernelOp::Loop;
        jLoop.var = vj;
        jLoop.begin = 0;
        jLoop.end = 2;
        jLoop.children.push_back(cid);
        jLoop.children.push_back(sid);
        const uint32_t jid = km.addNode(jLoop);
        KernelNode iLoop;
        iLoop.op = mlk::KernelOp::Loop;
        iLoop.var = vi;
        iLoop.begin = 0;
        iLoop.end = 2;
        // NOT parallel-marked -> serial root.
        iLoop.children.push_back(jid);
        (void)km.addNode(iLoop);
    }
    mlk::SlabEmitOptions opts;
    opts.sharedMemSlabs = true;
    auto src = mlk::emitCudaSource(km, symbols, opts);
    MLK_CHECK(src.has_value());
    if (!src.has_value()) {
        std::fprintf(stderr, "  emit: %s\n", src.error().message.c_str());
        return;
    }
    const std::string& s = *src;
    MLK_CHECK(s.find("__global__ void mlk_dev_0_sm(") !=
              std::string::npos);
    MLK_CHECK(s.find("extern __shared__ double mlk_smem[];") !=
              std::string::npos);
    MLK_CHECK(s.find("mlk_u < 4; mlk_u += mlk_bl") != std::string::npos);
    MLK_CHECK(s.find("__syncthreads();") != std::string::npos);
    // Serial twin: no live flag, no flat decomposition.
    MLK_CHECK(s.find("mlk_live") == std::string::npos);
    // Redirected read with the folded hull [0, 3].
    MLK_CHECK(s.find("mlk_smem[((2*") != std::string::npos);
    // The runtime pick is present for the serial root too.
    MLK_CHECK(s.find("mlk_dev_0_sm<<<mlk_grid, mlk_block, "
                     "mlk_smem_bytes>>>") != std::string::npos);
}

// The C harness around the EXTRACTED generated geometry helper: hand-
// computed assignments, the hardware-caps envelope, the soundness
// condition (multi-axis threads == total exactly; fallback threads >=
// total with the device's mlk_flat guard absorbing the ceil excess),
// and exhaustive hardware-index -> flat injectivity for small spaces
// (each padded instance realized by exactly one hardware thread).
// clang-format off
const char* kGeomHarnessPrelude = R"mlkgeom(#include <stdint.h>
#include <stdio.h>

static int mlk_fails = 0;

static void mlk_expect(const char* what, long long got, long long want) {
    if (got != want) {
        printf("HARNESS-FAIL %s: got %lld want %lld\n", what, got, want);
        ++mlk_fails;
    }
}

static int64_t mlk_prod(const int64_t* a, int n) {
    int64_t p = 1;
    for (int i = 0; i < n; ++i) p *= a[i];
    return p;
}
)mlkgeom";

const char* kGeomHarnessBody = R"mlkgeom(
static void mlk_rank_check(const char* name, const int64_t* g,
                           const int64_t* b, int64_t total) {
    const int64_t bprod = b[0] * b[1] * b[2];
    /* hardware axes, x fastest: tx, ty, tz (block dims), then
       bx, by, bz (grid dims) */
    const int64_t dims[6] = {b[0], b[1], b[2], g[0], g[1], g[2]};
    const int64_t space = mlk_prod(dims, 6);
    if (space != total) return; /* fallback shape: no bijection claim */
    if (space > 4194304) return; /* exhaustive only for small spaces */
    static unsigned char seen[4194304];
    for (int64_t i = 0; i < space; ++i) seen[i] = 0;
    int64_t idx[6] = {0, 0, 0, 0, 0, 0};
    for (int64_t rank = 0; rank < space; ++rank) {
        const int64_t flat =
            idx[5] * (g[1] * g[0] * bprod) + idx[4] * (g[0] * bprod) +
            idx[3] * bprod + idx[2] * (b[1] * b[0]) + idx[1] * b[0] +
            idx[0];
        if (flat != rank || seen[flat]) {
            printf("HARNESS-FAIL %s: flat %lld at rank %lld (disorder "
                   "or duplicate hardware thread)\n",
                   name, (long long)flat, (long long)rank);
            ++mlk_fails;
            return;
        }
        seen[flat] = 1;
        for (int p = 0; p < 6; ++p) {  /* x-fastest odometer */
            if (++idx[p] < dims[p]) break;
            idx[p] = 0;
        }
    }
}

static void mlk_check_case(const char* name, const int64_t* t, int n) {
    int64_t g[3] = {1, 1, 1};
    int64_t b[3] = {1, 1, 1};
    mlk_assign_geometry(t, n, g, b);
    const int64_t total = mlk_prod(t, n);
    const int64_t threads = mlk_prod(g, 3) * mlk_prod(b, 3);
    if (!(b[0] <= 1024 && b[1] <= 1024 && b[2] <= 64 &&
          b[0] * b[1] * b[2] <= 1024 && g[1] <= 65535 &&
          g[2] <= 65535)) {
        printf("HARNESS-FAIL %s: caps g=(%lld,%lld,%lld) "
               "b=(%lld,%lld,%lld)\n", name, (long long)g[0],
               (long long)g[1], (long long)g[2], (long long)b[0],
               (long long)b[1], (long long)b[2]);
        ++mlk_fails;
    }
    if (threads == total) {
        mlk_rank_check(name, g, b, total);
    } else if (threads > total) {
        /* the 1-D fallback: the ceil excess must keep the flat form */
        if (b[0] != 1024 || b[1] != 1 || b[2] != 1 || g[1] != 1 ||
            g[2] != 1) {
            printf("HARNESS-FAIL %s: unexpected fallback shape\n", name);
            ++mlk_fails;
        }
    } else {
        printf("HARNESS-FAIL %s: threads %lld < total %lld (lost "
               "coverage)\n", name, (long long)threads,
               (long long)total);
        ++mlk_fails;
    }
}

int main(void) {
    /* Hand-computed assignments (chain order outermost first). */
    {
        const int64_t t[4] = {8, 32, 8, 32}; /* [ti, i, tj, j] band */
        int64_t g[3] = {1, 1, 1}, b[3] = {1, 1, 1};
        mlk_assign_geometry(t, 4, g, b);
        /* block sweep packs j=32 then tj=8 into bx=256; i=32 and
           ti=8 overflow the 1024 cumulative cap onto gx=256 */
        mlk_expect("A g0", (long long)g[0], 256);
        mlk_expect("A g1", (long long)g[1], 1);
        mlk_expect("A g2", (long long)g[2], 1);
        mlk_expect("A b0", (long long)b[0], 256);
        mlk_expect("A b1", (long long)b[1], 1);
        mlk_expect("A b2", (long long)b[2], 1);
    }
    {
        const int64_t t[2] = {32, 32};
        int64_t g[3] = {1, 1, 1}, b[3] = {1, 1, 1};
        mlk_assign_geometry(t, 2, g, b);
        mlk_expect("B g0", (long long)g[0], 1);
        mlk_expect("B b0", (long long)b[0], 1024);
    }
    {
        const int64_t t[2] = {4, 5}; /* the test GEMM chain [i, j] */
        int64_t g[3] = {1, 1, 1}, b[3] = {1, 1, 1};
        mlk_assign_geometry(t, 2, g, b);
        mlk_expect("C g0", (long long)g[0], 1);
        mlk_expect("C b0", (long long)b[0], 20);
    }
    {
        const int64_t t[2] = {60000, 40000}; /* beyond the gy valve */
        int64_t g[3] = {1, 1, 1}, b[3] = {1, 1, 1};
        mlk_assign_geometry(t, 2, g, b);
        /* gx takes the inner 40000; the outer 60000 overflows gx's
           cumulative product? no: gx=40000, then 60000 needs
           40000*60000 > 2^31-1 on gx -> gy=60000 (<= 65535) */
        mlk_expect("D g0", (long long)g[0], 40000);
        mlk_expect("D g1", (long long)g[1], 60000);
        mlk_expect("D g2", (long long)g[2], 1);
        mlk_expect("D b0", (long long)b[0], 1);
    }
    {
        const int64_t t[2] = {70000, 70000}; /* no grid axis fits */
        int64_t g[3] = {1, 1, 1}, b[3] = {1, 1, 1};
        mlk_assign_geometry(t, 2, g, b);
        mlk_expect("E b0", (long long)b[0], 1024);
        mlk_expect("E g0", (long long)g[0], (70000LL * 70000LL + 1023) / 1024);
    }
    /* Envelope + coverage over a shape sweep. */
    {
        const int64_t shapes[][8] = {
            {4, 0, 0, 0, 0, 0, 0, 0},
            {4, 5, 0, 0, 0, 0, 0, 0},
            {32, 32, 0, 0, 0, 0, 0, 0},
            {8, 32, 8, 32, 0, 0, 0, 0},
            {2, 2, 3, 2, 0, 0, 0, 0},
            {3, 5, 7, 2, 4, 6, 0, 0},
            {1024, 1024, 0, 0, 0, 0, 0, 0},
            {1024, 1024, 1024, 0, 0, 0, 0, 0},
            {60000, 40000, 0, 0, 0, 0, 0, 0},
            {70000, 70000, 0, 0, 0, 0, 0, 0},
            {2, 0, 0, 0, 0, 0, 0, 0},
        };
        const int ns[] = {1, 2, 2, 4, 4, 6, 2, 3, 2, 2, 1};
        const char* names[] = {"s1",     "s2",     "s3",     "s4",
                               "s5",     "s6",     "s7",     "s8",
                               "s9",     "s10",    "s11"};
        const unsigned count =
            (unsigned)(sizeof(shapes) / sizeof(shapes[0]));
        for (unsigned i = 0; i < count; ++i) {
            mlk_check_case(names[i], shapes[i], ns[i]);
        }
    }
    if (mlk_fails == 0) printf("GEOM-HARNESS OK\n");
    return mlk_fails == 0 ? 0 : 1;
}
)mlkgeom";
// clang-format on

MLK_TEST(poly, cuda_emission_geometry_helper_behavioral) {
    // The geometry helper's generated text is plain C — compile THIS
    // exact text with cc (the same toolchain the cpp artifact path
    // probes) and verify its behavior natively:
    //   1. hand-computed assignments for representative shapes,
    //   2. the hardware-caps envelope on every returned dim,
    //   3. the SOUNDNESS condition: multi-axis launches satisfy
    //      threads == total exactly (never-clamp bijection onto the
    //      padded instance space), fallback launches threads >= total
    //      (the device's mlk_flat guard absorbs the ceil excess),
    //   4. for small multi-axis shapes, exhaustive injectivity of the
    //      hardware-index -> flat map (each instance executed once).
    // Without cc the test skips honestly (the skip condition IS the
    // recorded probe).
    if (!backendToolchainReady()) return;
    SymbolTable symbols;
    KernelModule base = buildGemmKernel(symbols);
    auto mod = buildScheduledGemm(symbols, base, 0);
    MLK_CHECK(mod.has_value());
    if (!mod.has_value()) return;
    auto src = mlk::emitCudaSource(*mod, symbols);
    MLK_CHECK(src.has_value());
    if (!src.has_value()) return;
    // Extract the helper text (marker to the first unindented close).
    const std::string marker = "static void mlk_assign_geometry(";
    const std::size_t begin = src->find(marker);
    MLK_CHECK(begin != std::string::npos);
    if (begin == std::string::npos) return;
    const std::size_t end = src->find("\n}\n", begin);
    MLK_CHECK(end != std::string::npos);
    if (end == std::string::npos) return;
    const std::string helper = src->substr(begin, end - begin + 3);

    // Run-unique workdir under the cwd (ctest's build tree — never
    // /tmp, matching the driver's own workdir policy).
    const std::string dir = std::string("./fk_cudaGeom_test-") +
                            std::to_string(static_cast<long long>(
                                std::chrono::steady_clock::now()
                                    .time_since_epoch()
                                    .count()));
    auto made = mlk::createDirs(dir);
    MLK_CHECK(made.has_value());
    if (!made.has_value()) return;
    const std::string harnessSrc = dir + "/harness.c";
    const std::string harnessBin = dir + "/harness";
    {
        FILE* f = std::fopen(harnessSrc.c_str(), "wb");
        MLK_CHECK(f != nullptr);
        if (f == nullptr) return;
        std::fprintf(f, "%s", kGeomHarnessPrelude);
        std::fprintf(f, "%s", helper.c_str());
        std::fprintf(f, "%s", kGeomHarnessBody);
        std::fclose(f);
    }
    const std::string cmd =
        std::string("cc -O2 -o ") + harnessBin + " " + harnessSrc +
        " 2> " + dir + "/cc.log";
    const int ccStatus = std::system(cmd.c_str());
    MLK_CHECK(ccStatus == 0);
    if (ccStatus != 0) {
        std::fprintf(stderr, "  harness compile failed; cc.log:\n");
        FILE* lg = std::fopen((dir + "/cc.log").c_str(), "rb");
        if (lg != nullptr) {
            char buf[512];
            std::size_t rd = std::fread(buf, 1, sizeof(buf), lg);
            std::fwrite(buf, 1, rd, stderr);
            std::fclose(lg);
        }
        return;
    }
    const int runStatus = std::system((harnessBin + " > " + dir +
                                       "/run.log 2>&1").c_str());
    if (runStatus != 0) {
        std::fprintf(stderr, "  harness run failed; run.log:\n");
        FILE* lg = std::fopen((dir + "/run.log").c_str(), "rb");
        if (lg != nullptr) {
            char buf[4096];
            std::size_t rd = std::fread(buf, 1, sizeof(buf), lg);
            std::fwrite(buf, 1, rd, stderr);
            std::fclose(lg);
        }
    }
    MLK_CHECK(runStatus == 0);
    // Self-cleanup (best effort — a leaked run-unique dir is noise,
    // not a failure): the logs stay on failure so the trace survives.
    if (runStatus == 0 && ccStatus == 0) {
        ::remove((dir + "/harness.c").c_str());
        ::remove(harnessBin.c_str());
        ::remove((dir + "/cc.log").c_str());
        ::remove((dir + "/run.log").c_str());
        ::rmdir(dir.c_str());
    }
}

MLK_TEST(poly, cuda_emission_policy_classification) {
    // The exactness boundary is OP-SET based, declared, never silent:
    // mul/add GEMM -> bit-exact; any device-libm transcendental
    // (exp, the softmax class) -> ULP-bounded. The classification the
    // gate consumes is the SAME scan the emitter's apply() performs.
    SymbolTable symbols;
    KernelModule base = buildGemmKernel(symbols);
    MLK_CHECK(mlk::cudaArtifactBitExactPolicy(base, symbols));
    // A module touching device libm exp: policy flips (conservative).
    KernelModule km;
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = SmallVector<int64_t, 4>{4};
    a.isInput = true;
    KernelBuffer y;
    y.name = symbols.intern("Y");
    y.dims = SmallVector<int64_t, 4>{4};
    y.isOutput = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufY = km.addBuffer(y);
    const SymbolId vi = symbols.intern("i");
    KernelNode loop;
    loop.op = mlk::KernelOp::Loop;
    loop.var = vi;
    loop.begin = 0;
    loop.end = mlk::constants::kKernelLoopDynamicBound;
    KernelNode compute;
    compute.op = mlk::KernelOp::Compute;
    compute.math = mlk::MathOp::Exp;  // legacy single-op form
    compute.bufferA = bufA;
    KernelNode store;
    store.op = mlk::KernelOp::Store;
    store.bufferOut = bufY;
    {
        const uint32_t cid = km.addNode(compute);
        const uint32_t sid = km.addNode(store);
        loop.children.push_back(cid);
        loop.children.push_back(sid);
        (void)km.addNode(loop);
    }
    MLK_CHECK(!mlk::cudaArtifactBitExactPolicy(km, symbols));
    auto src = mlk::emitCudaSource(km, symbols);
    MLK_CHECK(src.has_value());
    if (src.has_value()) {
        MLK_CHECK(src->find("Exactness policy: ULP-BOUNDED") !=
                  std::string::npos);
    }
}

MLK_TEST(poly, cuda_driver_honest_skip) {
    // The recorded-probe contract, three ways:
    //   1. buildKernelArtifact(kind=Cuda) is an InvalidArgument (GPU
    //      builds go through the GPU entry points — the arch flag must
    //      be declared, never defaulted),
    //   2. an invalid arch is an InvalidArgument,
    //   3. without nvcc the GPU build is an honest UnsupportedCapability
    //      naming the compiler (the skip condition IS the recorded
    //      check); WITH nvcc the same call must produce a loadable
    //      artifact that runs the GEMM bit-exact vs the walker.
    SymbolTable symbols;
    KernelModule base = buildGemmKernel(symbols);
    auto mod = buildScheduledGemm(symbols, base, 0);
    MLK_CHECK(mod.has_value());
    if (!mod.has_value()) return;
    mlk::BackendDriverConfig cfg;
    auto wrong = mlk::buildKernelArtifact(*mod, symbols,
                                          mlk::ArtifactKind::Cuda, cfg);
    MLK_CHECK(!wrong.has_value());
    if (wrong.has_value()) return;
    MLK_CHECK(wrong.error().code == mlk::ErrorCode::InvalidArgument);

    mlk::GpuBackendDriverConfig gcfg;
    gcfg.arch = "not-an-arch";
    auto badArch = mlk::buildGpuKernelArtifact(*mod, symbols, gcfg);
    MLK_CHECK(!badArch.has_value());
    if (badArch.has_value()) return;
    MLK_CHECK(badArch.error().code == mlk::ErrorCode::InvalidArgument);

    gcfg.arch = "sm_70";
    auto art = mlk::buildGpuKernelArtifact(*mod, symbols, gcfg);
    if (!cudaToolchainReady()) {
        // Honest skip: the structured error names the toolchain.
        MLK_CHECK(!art.has_value());
        if (art.has_value()) return;
        MLK_CHECK(art.error().code ==
                      mlk::ErrorCode::UnsupportedCapability ||
                  art.error().code == mlk::ErrorCode::InvalidArtifact);
        return;
    }
    // Live path (CUDA-equipped machine only): four-way bit-exact.
    MLK_CHECK(art.has_value());
    if (!art.has_value()) {
        std::fprintf(stderr, "  gpu build: %s\n",
                     art.error().message.c_str());
        return;
    }
    MLK_CHECK(art->isGpu());
    constexpr std::size_t nA = static_cast<std::size_t>(kTestM * kTestK);
    constexpr std::size_t nB = static_cast<std::size_t>(kTestK * kTestN);
    constexpr std::size_t nC = static_cast<std::size_t>(kTestM * kTestN);
    SmallVector<double, 8> bufA(nA), bufB(nB), outW(nC, 0.0), outG(nC, 0.0);
    for (std::size_t i = 0; i < nA; ++i) {
        bufA[i] = static_cast<double>(i % 7) * 0.25;
    }
    for (std::size_t i = 0; i < nB; ++i) {
        bufB[i] = static_cast<double>(i % 5) * 0.5;
    }
    auto bind = [&](SmallVector<double, 8>& outC) {
        mlk::KernelBufferBindings io;
        io.inputs.push_back(bufA.data());
        io.inputs.push_back(bufB.data());
        io.outputs.push_back(outC.data());
        io.elements = nC;
        return io;
    };
    auto rw = mlk::executeKernelOnBuffers(*mod, symbols, bind(outW),
                                          nullptr);
    MLK_CHECK(rw.has_value());
    if (!rw.has_value()) return;
    auto rg = art->run(*mod, symbols, bind(outG));
    MLK_CHECK(rg.has_value());
    if (!rg.has_value()) {
        std::fprintf(stderr, "  gpu run: %s\n",
                     rg.error().message.c_str());
        return;
    }
    for (std::size_t i = 0; i < nC; ++i) {
        MLK_CHECK(outW[i] == outG[i]);  // bit-exact (Rule 43)
    }
}

MLK_TEST(poly, cuda_emission_rejects_call_nodes) {
    // Call nodes must be lowered by poly.synth first (Rule 121) — the
    // CUDA emitter refuses them exactly like the C++/assembly emitters.
    SymbolTable symbols;
    KernelModule km;
    KernelBuffer a;
    a.name = symbols.intern("A");
    a.dims = SmallVector<int64_t, 4>{4, 4};
    a.isInput = true;
    KernelBuffer y;
    y.name = symbols.intern("Y");
    y.dims = SmallVector<int64_t, 4>{4, 4};
    y.isOutput = true;
    const uint32_t bufA = km.addBuffer(a);
    const uint32_t bufY = km.addBuffer(y);
    KernelNode call;
    call.op = mlk::KernelOp::Call;
    call.math = mlk::MathOp::MatMul;
    call.bufferA = bufA;
    call.bufferOut = bufY;
    (void)km.addNode(call);
    auto src = mlk::emitCudaSource(km, symbols);
    MLK_CHECK(!src.has_value());
    if (src.has_value()) return;
    MLK_CHECK(src.error().code == mlk::ErrorCode::UnsupportedCapability);
}

MLK_TEST_MAIN("poly")
