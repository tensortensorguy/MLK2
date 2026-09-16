// Dependence analysis implementation (see dependence.h).
#include "mlk/poly/dependence.h"

#include <utility>

#include "mlk/poly/checked.h"

namespace mlk::poly {

namespace {

/// Single-disjunct relation wrapper over the product space.
struct Relation {
    VarSpace space{};
    Polyhedron sys{};
};

[[nodiscard]] Result<Relation> makeRelation(uint32_t depth) {
    Relation r;
    r.space = VarSpace{depth * 2, 0};
    r.sys.space = r.space;
    return r;
}

/// Embeds a domain's box over the dim block [base, base + depth).
/// Domains from the extractor are single-disjunct boxes; their rows are
/// copied with the dim block offset. Deep-dim pins (== 0 equalities) come
/// along and are essential for init statements.
[[nodiscard]] Result<void> embedDomain(const Statement& s, uint32_t base,
                                       uint32_t totalDims,
                                       Polyhedron& sys) {
    if (s.domain.disjuncts.size() != 1) {
        return err(ErrorCode::InvalidArgument,
                   "dependence domains must be single-disjunct");
    }
    const Polyhedron& dom = s.domain.disjuncts[0];
    if (dom.isEmptyFlag) {
        return err(ErrorCode::InvalidArgument,
                   "dependence on an empty statement domain");
    }
    for (const ConstraintRow& r : dom.rows) {
        ConstraintRow shifted;
        shifted.isEquality = r.isEquality;
        shifted.constant = r.constant;
        shifted.coeffs = SmallVector<int64_t, 8>(totalDims, 0);
        for (uint32_t d = 0; d < s.space.nDims && d < r.coeffs.size(); ++d) {
            shifted.coeffs[base + d] = r.coeffs[d];
        }
        sys.addRow(std::move(shifted));
    }
    return {};
}

/// Adds the flat-index tie: flat_src(i) - flat_dst(j) == 0.
[[nodiscard]] Result<void> addTieRow(const MemoryAccess& src,
                                     const MemoryAccess& dst, uint32_t depth,
                                     Polyhedron& sys) {
    const AffineExpr& fs = src.flatIndex.outputs[0];
    const AffineExpr& fd = dst.flatIndex.outputs[0];
    ConstraintRow tie;
    tie.isEquality = true;
    tie.coeffs = SmallVector<int64_t, 8>(sys.space.totalVars(), 0);
    for (uint32_t d = 0; d < depth; ++d) {
        MLK_TRY_VAR(a, checked::subLimited(fs.coeffOf(d),
                                           fd.coeffOf(d)));
        // src coeff on i_d, minus dst coeff on j_d (same d index).
        tie.coeffs[d] = fs.coeffOf(d);
        tie.coeffs[depth + d] = -fd.coeffOf(d);
        (void)a;
    }
    MLK_TRY_VAR(k, checked::subLimited(fs.constant, fd.constant));
    tie.constant = -k;
    sys.addRow(std::move(tie));
    return {};
}

/// Builds the "sink >lex source" disjunction over the product space:
/// j_0 > i_0, or (equal prefix and j_1 > i_1), ... depth disjuncts.
[[nodiscard]] Result<PresburgerSet> lexGreater(uint32_t depth) {
    PresburgerSet out;
    out.space = VarSpace{depth * 2, 0};
    for (uint32_t pivot = 0; pivot < depth; ++pivot) {
        Polyhedron p;
        p.space = out.space;
        // Equalities on dims [0, pivot): j_d - i_d == 0.
        for (uint32_t d = 0; d < pivot; ++d) {
            ConstraintRow eq;
            eq.isEquality = true;
            eq.coeffs = SmallVector<int64_t, 8>(depth * 2, 0);
            eq.coeffs[d] = -1;
            eq.coeffs[depth + d] = 1;
            p.addRow(std::move(eq));
        }
        // j_pivot - i_pivot - 1 >= 0.
        ConstraintRow gt;
        gt.isEquality = false;
        gt.coeffs = SmallVector<int64_t, 8>(depth * 2, 0);
        gt.coeffs[pivot] = -1;
        gt.coeffs[depth + pivot] = 1;
        gt.constant = -1;
        p.addRow(std::move(gt));
        out.disjuncts.push_back(std::move(p));
    }
    return out;
}

/// "sink == source" (all dims equal) — a single conjunctive system.
[[nodiscard]] Result<PresburgerSet> lexEqual(uint32_t depth) {
    PresburgerSet out;
    out.space = VarSpace{depth * 2, 0};
    Polyhedron p;
    p.space = out.space;
    for (uint32_t d = 0; d < depth; ++d) {
        ConstraintRow eq;
        eq.isEquality = true;
        eq.coeffs = SmallVector<int64_t, 8>(depth * 2, 0);
        eq.coeffs[d] = -1;
        eq.coeffs[depth + d] = 1;
        p.addRow(std::move(eq));
    }
    out.disjuncts.push_back(std::move(p));
    return out;
}

}  // namespace

bool dependenceLive(const Dependence& dep) noexcept {
    // Stored dependences were filtered at construction; the flag reflects
    // their relation's liveness.
    return !dep.relation.disjuncts.empty() &&
           !dep.relation.disjuncts[0].isEmptyFlag;
}

Result<SmallVector<Dependence, 16>> computeDependences(const Scop& scop) {
    const uint32_t depth = scop.depth;
    if (depth == 0) {
        return err(ErrorCode::InvalidArgument,
                   "dependence analysis on a zero-depth SCoP");
    }
    SmallVector<Dependence, 16> out;
    MLK_TRY_VAR(lexGt, lexGreater(depth));
    MLK_TRY_VAR(lexEq, lexEqual(depth));

    for (const Statement& src : scop.statements) {
        for (const Statement& dst : scop.statements) {
            for (uint32_t ai = 0; ai < src.accesses.size(); ++ai) {
                const MemoryAccess& a = src.accesses[ai];
                if (dst.id != src.id && dst.id < src.id) {
                    // Statement pairs are visited in both directions only
                    // when the DATAFLOW direction demands it (checked by
                    // kind below); skip symmetric duplicates.
                    continue;
                }
                for (uint32_t bi = 0; bi < dst.accesses.size(); ++bi) {
                    const MemoryAccess& b = dst.accesses[bi];
                    if (a.bufferId != b.bufferId) continue;
                    DepKind kind;
                    if (a.isWrite && !b.isWrite) kind = DepKind::Raw;
                    else if (!a.isWrite && b.isWrite) kind = DepKind::War;
                    else if (a.isWrite && b.isWrite) kind = DepKind::Waw;
                    else continue;  // RAR: no dependence
                    // Original program order: source before sink.
                    if (src.id == dst.id) {
                        // Intra-statement: distinct instances only.
                        // (fall through with lexGreater purification)
                    } else if (dst.id < src.id) {
                        // Sink statement comes EARLIER in program order:
                        // a forward dependence is impossible; a backward
                        // (anti/output) dependence requires the sink to
                        // execute AFTER the source — that is exactly
                        // sink >lex source when the sink statement is
                        // later, which fails here. Skip: the pair (later
                        // src, earlier dst) can only produce a dep with
                        // kind reversed — handled when the pair is
                        // visited as (dst, src).
                        continue;
                    }

                    MLK_TRY_VAR(rel, makeRelation(depth));
                    MLK_TRYV(embedDomain(src, 0, depth * 2, rel.sys));
                    MLK_TRYV(embedDomain(dst, depth, depth * 2, rel.sys));
                    MLK_TRYV(addTieRow(a, b, depth, rel.sys));
                    if (rel.sys.isEmptyFlag) continue;
                    // Original-order purification.
                    //
                    // CHAIN-FINAL READS (softmax class): a RAW whose
                    // SOURCE write is an ACCUMULATE store (AccumMode
                    // Add/Max) feeds a reduction chain; a plain reader
                    // of that location consumes the CHAIN's FINAL value
                    // — in instance terms it may depend on EVERY chain
                    // writer, so the lexicographic "sink >=lex source"
                    // narrowing (which models per-iteration flow for
                    // init/accumulate pairs) would be UNSOUND here: it
                    // let the scheduler interleave reader and chain and
                    // emitted kernels reading the RUNNING reduction
                    // value. Keep the FULL tie relation for this class
                    // (a conservative superset of the true dataflow):
                    // any schedule row varying a chain dim then carries
                    // distances of both signs and is rejected, which
                    // forces band separation (or the honest "no
                    // schedule found" fallback) for chain-reading
                    // kernels.
                    PresburgerSet partial;
                    partial.space = rel.space;
                    partial.disjuncts.push_back(std::move(rel.sys));
                    const bool chainFinalRead =
                        kind == DepKind::Raw && src.id != dst.id &&
                        src.accum != AccumMode::None;
                    if (src.id == dst.id) {
                        MLK_TRY_VAR(purified,
                                    PresburgerSet::intersect(partial, lexGt));
                        MLK_TRY_VAR(f, purified.feasibility());
                        if (f == Feasibility::Empty) continue;
                        Dependence dep;
                        dep.srcStmt = src.id;
                        dep.dstStmt = dst.id;
                        dep.srcAccess = ai;
                        dep.dstAccess = bi;
                        dep.kind = kind;
                        dep.relation = std::move(purified);
                        if (out.size() >= kPolyMaxDependences) {
                            return err(ErrorCode::ResourceExhausted,
                                       "dependence budget exceeded");
                        }
                        out.push_back(std::move(dep));
                    } else if (chainFinalRead) {
                        // Full relation: every (chain writer, reader)
                        // instance pair is a potential dataflow edge.
                        MLK_TRY_VAR(f, partial.feasibility());
                        if (f == Feasibility::Empty) continue;
                        Dependence dep;
                        dep.srcStmt = src.id;
                        dep.dstStmt = dst.id;
                        dep.srcAccess = ai;
                        dep.dstAccess = bi;
                        dep.kind = kind;
                        dep.relation = std::move(partial);
                        if (out.size() >= kPolyMaxDependences) {
                            return err(ErrorCode::ResourceExhausted,
                                       "dependence budget exceeded");
                        }
                        out.push_back(std::move(dep));
                    } else {
                        // Cross-statement: sink >=lex source, with the
                        // equal-prefix case allowed only because the
                        // source statement precedes the sink statement
                        // in program order (dst.id > src.id holds).
                        MLK_TRY_VAR(eqPart,
                                    PresburgerSet::intersect(partial, lexEq));
                        MLK_TRY_VAR(gtPart,
                                    PresburgerSet::intersect(partial, lexGt));
                        MLK_TRY_VAR(ordered,
                                    PresburgerSet::unite(eqPart, gtPart));
                        MLK_TRY_VAR(f, ordered.feasibility());
                        if (f == Feasibility::Empty) continue;
                        Dependence dep;
                        dep.srcStmt = src.id;
                        dep.dstStmt = dst.id;
                        dep.srcAccess = ai;
                        dep.dstAccess = bi;
                        dep.kind = kind;
                        dep.relation = std::move(ordered);
                        if (out.size() >= kPolyMaxDependences) {
                            return err(ErrorCode::ResourceExhausted,
                                       "dependence budget exceeded");
                        }
                        out.push_back(std::move(dep));
                    }
                }
            }
        }
    }
    return out;
}

}  // namespace mlk::poly
