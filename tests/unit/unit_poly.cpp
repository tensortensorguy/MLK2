// Polyhedral engine unit tests (Rules 10, 33, 42, 90): exact rational
// arithmetic, affine expressions, Presburger set operations, FM emptiness,
// lexmin/lexmax witness verification, affine map image/preimage.
#include <optional>

#include "mlk/poly/poly.h"

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
    MLK_CHECK_EQ(s1.accesses.size(), 3);  // A, B, C
    MLK_CHECK(s0.accesses[0].isWrite);
    MLK_CHECK(!s1.accesses[0].isWrite);
    MLK_CHECK(!s1.accesses[1].isWrite);
    MLK_CHECK(s1.accesses[2].isWrite);
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

MLK_TEST_MAIN("poly")
