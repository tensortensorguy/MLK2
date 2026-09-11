// Schedule → Kernel IR code generation (see codegen.h).
//
// The emitter walks schedule levels recursively over the statement set:
//   - a level where every alive statement is CONSTANT (its theta value
//     does not vary with the remaining dims) is a SEPARATOR: statements
//     emit in (value, origOrder) order — statement-order at equal times;
//   - a level where alive statements vary in ONE dimension d (row == ±e_d
//     per statement; mixed dims bail) emits ONE loop over d (fusion);
//     statements with different dims at the same level bail (fission
//     beyond the synth class is not emitted);
//   - constant statements with value below the loop's lower bound emit
//     BEFORE the loop (pinned reduced dims: value 0 = the dim minimum);
//     values inside the range bail; values above emit after the loop;
//   - tiled band levels emit a TILE loop (constant bounds floor(lo/t),
//     floor(hi/t)) and a POINT loop with affine bounds t*ti and
//     t*ti + (t-1); divisibility of the box bounds by t is required so
//     the clipping is exact without piecewise forms.
// Payloads replay the statement's Compute chain (ElemIdx operands
// re-indexed from scop-dim coefficients to stack-position coefficients)
// and its Store (accumulate preserved).
#include "mlk/poly/codegen.h"

#include <algorithm>
#include <string>
#include <utility>

namespace mlk::poly {

namespace {

/// Returns the pinned constant value when statement s is a singleton at
/// dim d (its full-rank domain pins the dim), or false otherwise. Dims at
/// or beyond the statement's own depth are pinned to 0 by extraction.
[[nodiscard]] bool singletonAt(const Statement& s, uint32_t d,
                               int64_t* value) {
    if (d < s.ownLower.size() && d < s.ownUpper.size()) {
        if (s.ownLower[d] == s.ownUpper[d]) {
            *value = s.ownLower[d];
            return true;
        }
        return false;
    }
    if (d >= s.depth) {
        *value = 0;  // deep-dim pin (init-then-accumulate semantics)
        return true;
    }
    return false;
}

/// True when row r's form is constant for statement s (no dim coeffs).
[[nodiscard]] bool rowConstantFor(const Statement& s,
                                  const PolySchedule& sched, uint32_t r,
                                  int64_t* value) {
    const auto& coeffs = sched.rows[r].stmtCoeffs[s.id];
    for (uint32_t d = 0; d < sched.depth && d + 1 < coeffs.size(); ++d) {
        if (coeffs[d + 1] != 0) return false;
    }
    *value = coeffs[0];
    return true;
}

struct Emitter {
    const Scop* scop{nullptr};
    const PolySchedule* sched{nullptr};
    const TiledInfo* tiled{nullptr};
    const KernelModule* baseline{nullptr};
    SymbolTable* symbols{nullptr};
    KernelModule out{};
    // Stack position -> scop dim (-1 for tile-index loops whose variable
    // is not a scop dimension).
    SmallVector<int32_t, 8> dimAtStack{};
    SmallVector<uint32_t, 8> body{};
    SmallVector<bool, 8> emitted{};
    bool failed{false};
    int64_t copyBudget{1};
    std::string reason{};

    void fail(std::string why) {
        if (!failed) {
            failed = true;
            reason = std::move(why);
        }
    }

    /// Re-expresses per-dim coefficients in stack-position order.
    [[nodiscard]] SmallVector<int64_t, 4> mapCoeffs(
        const SmallVector<int64_t, 4>& dimCoeffs) const {
        SmallVector<int64_t, 4> mapped(dimAtStack.size(), 0);
        for (uint32_t p = 0; p < dimAtStack.size(); ++p) {
            const int32_t d = dimAtStack[p];
            if (d >= 0 && d < static_cast<int32_t>(dimCoeffs.size())) {
                mapped[p] = dimCoeffs[d];
            }
        }
        return mapped;
    }

    [[nodiscard]] KernelOperand mapOperand(const KernelOperand& o) const {
        KernelOperand mapped = o;
        if (o.kind == KernelOperand::Kind::ElemIdx) {
            mapped.idxCoeffs = mapCoeffs(o.idxCoeffs);
        }
        return mapped;
    }

    [[nodiscard]] Result<void> emitPayload(const Statement& s) {
        KernelNode compute;
        compute.op = KernelOp::Compute;
        for (const KernelExpr& e : s.exprs) {
            KernelExpr mapped;
            mapped.op = e.op;
            mapped.a = mapOperand(e.a);
            mapped.b = mapOperand(e.b);
            compute.exprs.push_back(std::move(mapped));
        }
        if (compute.exprs.empty()) {
            fail("statement without expressions");
            return {};
        }
        KernelNode store;
        store.op = KernelOp::Store;
        store.bufferOut = s.storeBuffer;
        store.accumulate = s.accumulate;
        store.outIndexCoeffs = mapCoeffs(s.storeCoeffs);
        store.outIndexOffset = s.storeOffset;
        const uint32_t cid = out.addNode(compute);
        const uint32_t sid = out.addNode(store);
        body.push_back(cid);
        body.push_back(sid);
        return {};
    }

    /// A constant statement may be hoisted before a loop at level r when
    /// ALL its remaining rows are constants that never exceed the minimum
    /// remaining value of any varying statement (conservative box check).
    [[nodiscard]] bool canEmitBefore(const Statement& s, uint32_t r) const {
        for (uint32_t r2 = r + 1; r2 < sched->rows.size(); ++r2) {
            int64_t sv = 0;
            if (!rowConstantFor(s, *sched, r2, &sv)) return false;
            for (const Statement& t : scop->statements) {
                if (t.id == s.id) continue;
                if (t.depth <= r2 && r2 >= t.depth) {
                    // t pinned at this deep dim: its remaining value is
                    // its constant coefficient (already counted) — skip.
                }
                // t's minimum for row r2 over its (box) domain.
                const auto& tc = sched->rows[r2].stmtCoeffs[t.id];
                int64_t tmin = tc[0];
                for (uint32_t d = 0; d < scop->depth && d + 1 < tc.size();
                     ++d) {
                    const int64_t c = tc[d + 1];
                    if (c == 0) continue;
                    const int64_t lo = d < t.ownLower.size()
                                           ? t.ownLower[d]
                                           : 0;
                    const int64_t hi = d < t.ownUpper.size()
                                           ? t.ownUpper[d]
                                           : 0;
                    tmin += c * (c > 0 ? lo : hi);
                }
                if (sv > tmin) return false;
            }
        }
        return true;
    }

    /// Emits payloads for (value, origOrder)-sorted constant items.
    Result<void> emitConstItems(
        SmallVector<std::pair<int64_t, uint32_t>, 8> items) {
        std::sort(items.begin(), items.end(),
                  [&](const auto& a, const auto& b) {
                      if (a.first != b.first) return a.first < b.first;
                      return scop->statements[a.second].origOrder <
                             scop->statements[b.second].origOrder;
                  });
        for (const auto& item : items) {
            MLK_TRYV(emitPayload(scop->statements[item.second]));
            emitted[item.second] = true;
        }
        return {};
    }

    /// Emits a loop over dim `varDim` (tiled when in the band) whose body
    /// is the recursion at level r+1 for `varStmts`.
    [[nodiscard]] Result<void> emitLoop(SmallVector<uint32_t, 8> varStmts,
                                        uint32_t varDim, int64_t lo,
                                        int64_t hi, uint32_t r) {
        const bool tiledHere = tiled->tiled && r < tiled->bandEnd;
        const int64_t tileSize = tiledHere ? tiled->tileSizes[r] : 0;
        SymbolId varName =
            static_cast<std::size_t>(varDim) < scop->dimVars.size()
                ? scop->dimVars[varDim]
                : kInvalidSymbolId;
        if (varName == kInvalidSymbolId) {
            varName = symbols->intern("d" + std::to_string(varDim));
        }
        if (!tiledHere) {
            KernelNode loop;
            loop.op = KernelOp::Loop;
            loop.var = varName;
            loop.begin = lo;
            loop.end = hi + 1;  // executor form: var < end
            dimAtStack.push_back(static_cast<int32_t>(varDim));
            const SmallVector<uint32_t, 8> saved = body;
            body.clear();
            MLK_TRYV(emitLevel(std::move(varStmts), r + 1));
            loop.children.clear();
            for (const uint32_t c : body) loop.children.push_back(c);
            body = saved;
            dimAtStack.pop_back();
            const uint32_t lid = out.addNode(loop);
            body.push_back(lid);
            return {};
        }
        // Tiled: the tile loop iterates floor(v / t); the POINT loop is
        // affine per tile. Full tiles carry the affine bounds
        // [t*ti, t*ti + t - 1]; a trailing PARTIAL tile (when
        // (hi+1) % t != 0) carries [t*ti, hi] (constant inclusive end).
        // The inner subtree duplicates once per part (bounded by
        // kPolyMaxCodegenCopies across the nest). lo % t == 0 is required
        // (synth-kernel box bounds start at 0).
        if (lo % tileSize != 0) {
            fail("codegen: tile lower bound not aligned");
            return {};
        }
        const int64_t firstTile = lo / tileSize;
        const int64_t lastTile = hi / tileSize;
        const int64_t fullLast = (hi + 1) / tileSize - 1;
        const bool hasFull = fullLast >= firstTile;
        const bool hasPartial = (hi + 1) % tileSize != 0;
        if (hasPartial) {
            copyBudget *= 2;
            if (copyBudget > constants::kPolyMaxCodegenCopies) {
                fail("codegen: tile duplication exceeds copy budget");
                return {};
            }
        }
        if (hasFull) {
            KernelNode tile;
            tile.op = KernelOp::Loop;
            tile.var = symbols->intern(
                std::string(symbols->text(varName)) + std::string("_t"));
            tile.begin = firstTile;
            tile.end = fullLast + 1;
            const uint32_t tileId = out.addNode(tile);
            dimAtStack.push_back(-1);  // tile index: not a scop dim
            MLK_TRYV(emitPointBody(varStmts, varDim, r, tileSize, -1,
                                   tileId));
        }
        if (hasPartial) {
            KernelNode tile;
            tile.op = KernelOp::Loop;
            tile.var = symbols->intern(
                std::string(symbols->text(varName)) + std::string("_t"));
            tile.begin = lastTile;
            tile.end = lastTile + 1;
            const uint32_t tileId = out.addNode(tile);
            dimAtStack.push_back(-1);
            MLK_TRYV(emitPointBody(varStmts, varDim, r, tileSize, hi,
                                   tileId));
        }
        return {};
    }

    /// Emits the point loop + inner recursion for one tile part and wires
    /// it under the tile node `tileId` (which is already in `out`, with
    /// its stack slot pushed). `partialEnd >= 0` selects the constant
    /// inclusive end (partial tile); otherwise the affine end
    /// t*ti + t - 1 is used.
    [[nodiscard]] Result<void> emitPointBody(
        SmallVector<uint32_t, 8>& varStmts, uint32_t varDim, uint32_t r,
        int64_t tileSize, int64_t partialEnd, uint32_t tileId) {
        SymbolId varName =
            static_cast<std::size_t>(varDim) < scop->dimVars.size()
                ? scop->dimVars[varDim]
                : kInvalidSymbolId;
        if (varName == kInvalidSymbolId) {
            varName = symbols->intern("d" + std::to_string(varDim));
        }
        KernelNode point;
        point.op = KernelOp::Loop;
        point.var = varName;
        point.beginCoeffs = SmallVector<int64_t, 4>(1, tileSize);
        point.beginOffset = 0;
        if (partialEnd >= 0) {
            point.end = partialEnd + 1;  // executor form: var < end
        } else {
            point.endCoeffs = SmallVector<int64_t, 4>(1, tileSize);
            point.endOffset = tileSize - 1;  // inclusive: t*ti + t - 1
        }
        dimAtStack.push_back(static_cast<int32_t>(varDim));
        const SmallVector<uint32_t, 8> saved = body;
        body.clear();
        MLK_TRYV(emitLevel(std::move(varStmts), r + 1));
        point.children.clear();
        for (const uint32_t c : body) point.children.push_back(c);
        body = saved;
        dimAtStack.pop_back();
        const uint32_t pointId = out.addNode(point);
        out.nodes[tileId].children.push_back(pointId);
        dimAtStack.pop_back();  // tile slot
        body.push_back(tileId);
        return {};
    }

    Result<void> emitLevel(SmallVector<uint32_t, 8> stmts, uint32_t r) {
        if (failed) return {};
        SmallVector<uint32_t, 8> alive;
        for (const uint32_t s : stmts) {
            if (!emitted[s]) alive.push_back(s);
        }
        if (alive.empty()) return {};
        if (r >= sched->rows.size()) {
            // Schedule exhausted: remaining statements share the same
            // time — original order (determinism, Rule 53).
            for (const uint32_t s : alive) {
                MLK_TRYV(emitPayload(scop->statements[s]));
                emitted[s] = true;
            }
            return {};
        }
        // Classify alive statements at this level.
        SmallVector<std::pair<int64_t, uint32_t>, 8> constItems;
        SmallVector<uint32_t, 8> varStmts;
        int32_t varDim = -1;
        for (const uint32_t s : alive) {
            const auto& coeffs = sched->rows[r].stmtCoeffs[s];
            int32_t varDimS = -1;
            bool varNegS = false;
            for (uint32_t d = 0; d < scop->depth && d + 1 < coeffs.size();
                 ++d) {
                const int64_t c = coeffs[d + 1];
                if (c == 0) continue;
                if (c != 1 && c != -1) {
                    fail("codegen: non-unimodular schedule row");
                    return {};
                }
                if (varDimS != -1) {
                    fail("codegen: schedule row combines dimensions");
                    return {};
                }
                varDimS = static_cast<int32_t>(d);
                varNegS = c < 0;
            }
            if (varDimS == -1) {
                constItems.emplace_back(coeffs[0], s);
            } else if (true) {
                // Singleton at this dim? The statement acts as a constant
                // item with its pinned value (pinned reduced dims).
                int64_t pinned = 0;
                if (singletonAt(scop->statements[s],
                                static_cast<uint32_t>(varDimS), &pinned)) {
                    constItems.emplace_back(coeffs[0] +
                                                coeffs[varDimS + 1] * pinned,
                                            s);
                } else {
                    if (varDim != -1 && varDim != varDimS) {
                        fail("codegen: fission across dims at one level");
                        return {};
                    }
                    varDim = varDimS;
                    varStmts.push_back(s);
                    (void)varNegS;
                }
            } else {
                constItems.emplace_back(coeffs[0], s);
            }
        }
        if (varStmts.empty()) {
            MLK_TRYV(emitConstItems(std::move(constItems)));
            return {};
        }
        // Loop bounds: intersection of the varying statements' boxes.
        int64_t lo = INT64_MIN;
        int64_t hi = INT64_MAX;
        for (const uint32_t s : varStmts) {
            const Statement& st = scop->statements[s];
            if (varDim >= static_cast<int32_t>(st.ownLower.size())) {
                fail("codegen: statement lacks bounds for the loop dim");
                return {};
            }
            lo = st.ownLower[varDim] > lo ? st.ownLower[varDim] : lo;
            hi = st.ownUpper[varDim] < hi ? st.ownUpper[varDim] : hi;
        }
        if (lo == INT64_MIN || hi == INT64_MAX || lo > hi) {
            fail("codegen: empty or unbounded loop range");
            return {};
        }
        // Constant statements around the loop range.
        SmallVector<std::pair<int64_t, uint32_t>, 8> before, after;
        for (const auto& item : constItems) {
            if (item.first < lo) {
                before.push_back(item);
            } else if (item.first > hi) {
                after.push_back(item);
            } else if (item.first == lo &&
                       canEmitBefore(scop->statements[item.second], r)) {
                // Same time as the first iteration, but every REMAINING
                // schedule row of the statement is a constant that does
                // not exceed the varying statements' minimums — emitting
                // before the loop preserves the lexicographic order.
                before.push_back(item);
            } else {
                fail("codegen: constant statement inside the loop range");
                return {};
            }
        }
        MLK_TRYV(emitConstItems(std::move(before)));
        MLK_TRYV(emitLoop(std::move(varStmts), static_cast<uint32_t>(varDim),
                          lo, hi, r));
        if (failed) return {};
        MLK_TRYV(emitConstItems(std::move(after)));
        return {};
    }
};

}  // namespace

Result<KernelModule> emitScheduledKernel(
    const Scop& scop, const PolySchedule& sched, const TiledInfo& tiled,
    const KernelModule& baseline, SymbolTable& symbols) {
    if (scop.statements.empty() || sched.nStmts != scop.statements.size()) {
        return err(ErrorCode::InvalidArgument, "codegen: shape mismatch");
    }
    Emitter em;
    em.scop = &scop;
    em.sched = &sched;
    em.tiled = &tiled;
    em.baseline = &baseline;
    em.symbols = &symbols;
    em.emitted = SmallVector<bool, 8>(scop.statements.size(), false);
    em.out.name = baseline.name;
    em.out.buffers = baseline.buffers;
    em.out.scheduleParams = baseline.scheduleParams;

    SmallVector<uint32_t, 8> all;
    for (uint32_t s = 0; s < scop.statements.size(); ++s) all.push_back(s);
    MLK_TRYV(em.emitLevel(std::move(all), 0));
    if (em.failed) {
        return err(ErrorCode::UnsupportedCapability,
                   "codegen: " + em.reason);
    }
    // Unreferenced leftovers (statements the schedule never emitted) are a
    // contract bug — refuse rather than drop computation silently.
    for (const bool e : em.emitted) {
        if (!e) {
            return err(ErrorCode::Internal,
                       "codegen: statement not emitted by the schedule");
        }
    }
    return std::move(em.out);
}

}  // namespace mlk::poly
