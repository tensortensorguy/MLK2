// Schedule → Kernel IR code generation (see codegen.h).
//
// The emitter walks schedule levels recursively over the statement set.
// Every level carries the pivot dim chosen by the scheduler (rows[i] /
// pivotDim[i]); skew coefficients on spent (outer) dims are legal in the
// schedule and intentionally invisible here: with the loop variable kept
// as the pivot dim and box bounds, the nest enumerates instances in
// exactly the schedule's lexicographic order.
//   - a level where every alive statement is CONSTANT (all nonzero row
//     coefficients sit on dims pinned for that statement) is a
//     SEPARATOR: statements emit in (value, origOrder) order —
//     statement-order at equal times;
//   - a level where alive statements vary emit ONE loop over the pivot
//     dim (fusion); statements with a non-positive pivot coefficient
//     bail (the scheduler's shape contract forbids them);
//   - constant statements with value below the loop's lower bound emit
//     BEFORE the loop (pinned reduced dims fold into the value); values
//     above emit after the loop; values inside the range RE-ENTER via
//     the piecewise split: the loop range is cut at every re-entry
//     value and the statement joins only the singleton segment [v, v]
//     (identical instance order to a runtime guard, but the hot
//     segments carry no branch; the Guard form remains as the budget
//     fallback);
//   - tiled band levels emit a TILE loop (constant bounds floor(lo/t),
//     floor(hi/t)) and a POINT loop with affine bounds t*ti and
//     t*ti + (t-1); divisibility of the box bounds by t is required so
//     the clipping is exact without piecewise forms.
// Payloads replay the statement's Compute chain (ElemIdx operands
// re-indexed from scop-dim coefficients to stack-position coefficients;
// coefficients on PINNED dims fold into the constant offset) and its
// Store (accumulate preserved).
#include "mlk/poly/codegen.h"

#include <algorithm>
#include <string>
#include <utility>

#include "schedule_common.h"

namespace mlk::poly {

namespace {


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
    // Per statement: pending guard conditions (scop dim, folded row
    // value), outermost first. Appended when the statement re-enters a
    // loop under a CLAST guard; materialized as nested Guard nodes at
    // EVERY payload emission (tile part duplication re-runs the
    // recursion, so the guards are never consumed). The referenced dims
    // are enclosing loop variables at every emission point.
    SmallVector<SmallVector<std::pair<uint32_t, int64_t>, 4>, 8>
        pendingGuards{};
    bool failed{false};
    int64_t copyBudget{1};
    std::string reason{};

    void fail(std::string why) {
        if (!failed) {
            failed = true;
            reason = std::move(why);
        }
    }

    /// Re-expresses per-dim coefficients in stack-position order, folding
    /// coefficients of PINNED (off-stack) dims into *outOffset. A
    /// non-pinned off-stack dim with a nonzero coefficient is a contract
    /// bug (the schedule would collapse varying instances) — fails.
    [[nodiscard]] Result<void> mapCoeffs(const Statement& s,
                                         const SmallVector<int64_t, 4>& dimCoeffs,
                                         SmallVector<int64_t, 4>& outStack,
                                         int64_t* outOffset) {
        outStack.clear();
        for (std::size_t i = 0; i < dimAtStack.size(); ++i) {
            outStack.push_back(0);
        }
        int64_t offset = 0;
        for (uint32_t d = 0; d < dimCoeffs.size(); ++d) {
            const int64_t c = dimCoeffs[d];
            if (c == 0) continue;
            int32_t pos = -1;
            for (uint32_t q = 0; q < dimAtStack.size(); ++q) {
                if (dimAtStack[q] == static_cast<int32_t>(d)) {
                    pos = static_cast<int32_t>(q);
                    break;
                }
            }
            if (pos >= 0) {
                outStack[static_cast<uint32_t>(pos)] = c;
                continue;
            }
            int64_t pin = 0;
            if (!detail::dimPinnedAt(s, d, &pin)) {
                fail("codegen: access references an unpinned unspent dim");
                *outOffset = 0;
                return {};
            }
            offset += c * pin;
        }
        *outOffset = offset;
        return {};
    }

    [[nodiscard]] Result<KernelOperand> mapOperand(const Statement& s,
                                                   const KernelOperand& o) {
        KernelOperand mapped = o;
        if (o.kind == KernelOperand::Kind::ElemIdx) {
            SmallVector<int64_t, 4> stack;
            int64_t off = 0;
            MLK_TRYV(mapCoeffs(s, o.idxCoeffs, stack, &off));
            mapped.idxCoeffs = std::move(stack);
            MLK_TRY_VAR(total, checked::addLimited(o.idxOffset, off));
            mapped.idxOffset = total;
        }
        return mapped;
    }

    [[nodiscard]] Result<void> emitPayload(const Statement& s) {
        KernelNode compute;
        compute.op = KernelOp::Compute;
        for (const KernelExpr& e : s.exprs) {
            KernelExpr mapped;
            mapped.op = e.op;
            MLK_TRY_VAR(a, mapOperand(s, e.a));
            mapped.a = std::move(a);
            MLK_TRY_VAR(b, mapOperand(s, e.b));
            mapped.b = std::move(b);
            compute.exprs.push_back(std::move(mapped));
        }
        if (compute.exprs.empty()) {
            fail("statement without expressions");
            return {};
        }
        KernelNode store;
        store.op = KernelOp::Store;
        store.bufferOut = s.storeBuffer;
        store.accum = s.accum;
        SmallVector<int64_t, 4> stack;
        int64_t off = 0;
        MLK_TRYV(mapCoeffs(s, s.storeCoeffs, stack, &off));
        store.outIndexCoeffs = std::move(stack);
        MLK_TRY_VAR(storeOff, checked::addLimited(s.storeOffset, off));
        store.outIndexOffset = storeOff;
        const uint32_t cid = out.addNode(compute);
        const uint32_t sid = out.addNode(store);
        // Pending guards nest around the payload (outermost condition
        // outermost). The condition "scop dim == value" becomes the
        // affine form 1*var[pos] - value == 0 over the CURRENT var
        // stack; the dim must be bound here (its loop encloses every
        // emission point of the statement).
        SmallVector<uint32_t, 4> wrapped;
        wrapped.push_back(cid);
        wrapped.push_back(sid);
        const auto& gs = pendingGuards[s.id];
        for (std::size_t g = gs.size(); g-- > 0;) {
            int32_t pos = -1;
            for (uint32_t q = 0; q < dimAtStack.size(); ++q) {
                if (dimAtStack[q] == static_cast<int32_t>(gs[g].first)) {
                    pos = static_cast<int32_t>(q);
                    break;
                }
            }
            if (pos < 0) {
                fail("codegen: guard dim is not bound at emission");
                return {};
            }
            KernelNode guard;
            guard.op = KernelOp::Guard;
            guard.guardCoeffs = SmallVector<int64_t, 4>(
                static_cast<std::size_t>(pos) + 1, 0);
            guard.guardCoeffs[static_cast<uint32_t>(pos)] = 1;
            guard.guardOffset = -gs[g].second;
            guard.children.clear();
            for (const uint32_t c : wrapped) guard.children.push_back(c);
            const uint32_t gid = out.addNode(guard);
            wrapped.clear();
            wrapped.push_back(gid);
        }
        for (const uint32_t c : wrapped) body.push_back(c);
        return {};
    }

    /// A constant statement may be hoisted before a loop at level r when
    /// ALL its remaining rows are constants (pin-folded) that never exceed
    /// the minimum remaining value of any varying statement (conservative
    /// box check).
    [[nodiscard]] bool canEmitBefore(const Statement& s, uint32_t r) const {
        for (uint32_t r2 = r + 1; r2 < sched->rows.size(); ++r2) {
            int64_t sv = 0;
            if (!detail::rowEffectiveConst(s, sched->rows[r2], &sv)) {
                return false;
            }
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

    /// Emits one plain (untiled) loop over dim `varDim` with constant
    /// inclusive bounds [lo, hi] whose body is the recursion at level
    /// r+1 for `varStmts`. Shared by the guarded fallback path and the
    /// piecewise split segments (identical marks: parallel/vector hints
    /// come from the schedule row).
    [[nodiscard]] Result<void> emitPlainLoop(
        const SmallVector<uint32_t, 8>& varStmts, uint32_t varDim,
        int64_t lo, int64_t hi, uint32_t r) {
        const bool rowParallel = r < sched->parallel.size()
                                     ? sched->parallel[r]
                                     : false;
        const bool rowVector = r < sched->vectorizable.size()
                                   ? sched->vectorizable[r]
                                   : false;
        SymbolId varName =
            static_cast<std::size_t>(varDim) < scop->dimVars.size()
                ? scop->dimVars[varDim]
                : kInvalidSymbolId;
        if (varName == kInvalidSymbolId) {
            varName = symbols->intern("d" + std::to_string(varDim));
        }
        KernelNode loop;
        loop.op = KernelOp::Loop;
        loop.var = varName;
        loop.begin = lo;
        loop.end = hi + 1;  // executor form: var < end
        loop.parallel = rowParallel;
        loop.vectorHint = rowVector;
        dimAtStack.push_back(static_cast<int32_t>(varDim));
        const SmallVector<uint32_t, 8> saved = body;
        body.clear();
        MLK_TRYV(emitLevel(varStmts, r + 1));
        loop.children.clear();
        for (const uint32_t c : body) loop.children.push_back(c);
        body = saved;
        dimAtStack.pop_back();
        const uint32_t lid = out.addNode(loop);
        body.push_back(lid);
        return {};
    }

    /// Emits a loop over dim `varDim` (tiled when in the band) whose body
    /// is the recursion at level r+1 for `varStmts`. The loop carries the
    /// scheduler's parallel/vector marks for its row: `parallel` means
    /// every dependence distance is identically zero at this level (the
    /// executor may thread disjoint index chunks); `vectorHint` marks the
    /// innermost enumeration of a SIMD-able parallel row.
    [[nodiscard]] Result<void> emitLoop(SmallVector<uint32_t, 8> varStmts,
                                        uint32_t varDim, int64_t lo,
                                        int64_t hi, uint32_t r) {
        const bool rowParallel = r < sched->parallel.size()
                                     ? sched->parallel[r]
                                     : false;
        const bool rowVector = r < sched->vectorizable.size()
                                   ? sched->vectorizable[r]
                                   : false;
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
            MLK_TRYV(emitPlainLoop(varStmts, varDim, lo, hi, r));
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
        // Tile parts are SIBLING instance ranges: each part replays the
        // same statements for its own point range. Snapshot the emission
        // marks and guard state before the first part and restore them
        // before the second, so the partial part re-emits everything the
        // full part emitted (statements hoisted before the tile loop
        // were emitted before the snapshot and stay filtered).
        const SmallVector<bool, 8> emittedSnapshot = emitted;
        const SmallVector<SmallVector<std::pair<uint32_t, int64_t>, 4>, 8>
            guardsSnapshot = pendingGuards;
        if (hasFull) {
            KernelNode tile;
            tile.op = KernelOp::Loop;
            tile.var = symbols->intern(
                std::string(symbols->text(varName)) + std::string("_t"));
            tile.begin = firstTile;
            tile.end = fullLast + 1;
            tile.parallel = rowParallel;  // dist-0 rows: tiles are disjoint
            const uint32_t tileId = out.addNode(tile);
            dimAtStack.push_back(-1);  // tile index: not a scop dim
            MLK_TRYV(emitPointBody(varStmts, varDim, r, tileSize, -1,
                                   tileId, rowParallel, rowVector));
        }
        if (hasPartial) {
            emitted = emittedSnapshot;
            pendingGuards = guardsSnapshot;
            KernelNode tile;
            tile.op = KernelOp::Loop;
            tile.var = symbols->intern(
                std::string(symbols->text(varName)) + std::string("_t"));
            tile.begin = lastTile;
            tile.end = lastTile + 1;
            tile.parallel = rowParallel;
            const uint32_t tileId = out.addNode(tile);
            dimAtStack.push_back(-1);
            MLK_TRYV(emitPointBody(varStmts, varDim, r, tileSize, hi,
                                   tileId, rowParallel, rowVector));
        }
        return {};
    }

    /// Emits the point loop + inner recursion for one tile part and wires
    /// it under the tile node `tileId` (which is already in `out`, with
    /// its stack slot pushed). `partialEnd >= 0` selects the constant
    /// inclusive end (partial tile); otherwise the affine end
    /// t*ti + t - 1 is used. `rowParallel`/`rowVector` are the schedule
    /// row's marks (tile and point loops share the row's parallelism;
    /// the SIMD hint applies to the point enumeration).
    [[nodiscard]] Result<void> emitPointBody(
        const SmallVector<uint32_t, 8>& varStmts, uint32_t varDim,
        uint32_t r, int64_t tileSize, int64_t partialEnd, uint32_t tileId,
        bool rowParallel, bool rowVector) {
        SymbolId varName =
            static_cast<std::size_t>(varDim) < scop->dimVars.size()
                ? scop->dimVars[varDim]
                : kInvalidSymbolId;
        if (varName == kInvalidSymbolId) {
            varName = symbols->intern("d" + std::to_string(varDim));
        }
        // The tile index sits in the -1 stack slot pushed by emitLoop.
        // Affine bound coefficients are STACK-ABSOLUTE (the walker
        // indexes coeffs[p] by vars[p], outermost first), so the tile
        // coefficient goes at that slot with zeros elsewhere — a short
        // [t] vector would multiply the OUTERMOST var (a latent
        // round-6 bug masked by single-tile test ranges).
        const std::size_t tilePos = dimAtStack.size() - 1;
        KernelNode point;
        point.op = KernelOp::Loop;
        point.var = varName;
        point.beginCoeffs = SmallVector<int64_t, 4>(tilePos + 1, 0);
        point.beginCoeffs[tilePos] = tileSize;
        point.beginOffset = 0;
        point.parallel = rowParallel;
        point.vectorHint = rowVector;  // SIMD-able innermost enumeration
        if (partialEnd >= 0) {
            point.end = partialEnd + 1;  // executor form: var < end
        } else {
            point.endCoeffs = SmallVector<int64_t, 4>(tilePos + 1, 0);
            point.endCoeffs[tilePos] = tileSize;
            point.endOffset = tileSize - 1;  // inclusive: t*ti + t - 1
        }
        dimAtStack.push_back(static_cast<int32_t>(varDim));
        const SmallVector<uint32_t, 8> saved = body;
        body.clear();
        // COPY, not move: the sibling tile part replays the same
        // statement list (a moved-from list silently empties the
        // partial part's recursion).
        MLK_TRYV(emitLevel(varStmts, r + 1));
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

    // --- Piecewise split (CLAST range splitting) ----------------------
    //
    // A row-constant statement re-entering a loop is realized today by a
    // per-iteration affine guard "pivot == v". The split realizes the
    // SAME instance order STRUCTURALLY: the loop range is cut at every
    // guard value; the statement participates only in the singleton
    // segment [v, v], where the guard condition holds for every
    // iteration and the payload emits unguarded. Tiled levels cut the
    // TILE range at floor(v / t); the singleton tile emits constant
    // point segments. Execution order is provably identical to the
    // guarded form (segments enumerate in ascending pivot order; the
    // deeper recursion is unchanged), so the transformation is bit-exact
    // by construction — and the hot segments lose the branch entirely.
    // Fallback: when the replay budget would be exceeded the level falls
    // back to the guard form (never fails the kernel).

    /// One replayed body at a split level: a plain loop over the pivot
    /// range (untiled levels), or one tile sub-range with either the
    /// standard point body or explicit constant point segments (tiled
    /// levels, singleton tile only).
    struct SplitBody {
        int64_t lo{0};
        int64_t hi{0};  // inclusive (pivot range or tile range)
        bool tiled{false};
        int64_t partialEnd{-1};  // tiled: constant point end of the LAST tile
        SmallVector<uint32_t, 8> stmts{};
        SmallVector<std::pair<int64_t, int64_t>, 4> segRange{};
        SmallVector<SmallVector<uint32_t, 8>, 4> segStmts{};
    };

    [[nodiscard]] Result<void> emitSplitBody(const SplitBody& b,
                                             uint32_t varDim, uint32_t r) {
        if (!b.tiled) {
            MLK_TRYV(emitPlainLoop(b.stmts, varDim, b.lo, b.hi, r));
            return {};
        }
        const bool rowParallel = r < sched->parallel.size()
                                     ? sched->parallel[r]
                                     : false;
        const bool rowVector = r < sched->vectorizable.size()
                                   ? sched->vectorizable[r]
                                   : false;
        const int64_t tileSize = tiled->tileSizes[r];
        SymbolId varName =
            static_cast<std::size_t>(varDim) < scop->dimVars.size()
                ? scop->dimVars[varDim]
                : kInvalidSymbolId;
        if (varName == kInvalidSymbolId) {
            varName = symbols->intern("d" + std::to_string(varDim));
        }
        KernelNode tile;
        tile.op = KernelOp::Loop;
        tile.var = symbols->intern(
            std::string(symbols->text(varName)) + std::string("_t"));
        tile.begin = b.lo;
        tile.end = b.hi + 1;
        tile.parallel = rowParallel;  // dist-0 rows: tiles are disjoint
        const uint32_t tileId = out.addNode(tile);
        dimAtStack.push_back(-1);  // tile index: not a scop dim
        if (b.segRange.empty()) {
            // Standard tile body (affine point bounds or the partial
            // constant end) — the proven emitPointBody path.
            MLK_TRYV(emitPointBody(b.stmts, varDim, r, tileSize,
                                   b.partialEnd, tileId, rowParallel,
                                   rowVector));
            return {};
        }
        // Singleton tile: constant point segments (the tile var is fixed
        // at b.lo == b.hi, so the affine bound degenerates to a constant
        // and nothing references the tile slot). Segments replay the
        // deeper recursion — snapshot/restore between them.
        const SmallVector<bool, 8> snapE = emitted;
        const SmallVector<SmallVector<std::pair<uint32_t, int64_t>, 4>, 8>
            snapG = pendingGuards;
        for (std::size_t si = 0; si < b.segRange.size(); ++si) {
            if (si > 0) {
                // Restore replay marks ONLY for statements this segment
                // replays: payloads emitted in earlier segments alone
                // (single-segment re-entries) keep their marks so they
                // are never re-emitted here. Guards are restored
                // wholesale (deeper fallback levels re-push them).
                for (const uint32_t s : b.segStmts[si]) {
                    emitted[s] = snapE[s];
                }
                pendingGuards = snapG;
            }
            KernelNode point;
            point.op = KernelOp::Loop;
            point.var = varName;
            point.begin = b.segRange[si].first;
            point.end = b.segRange[si].second + 1;  // executor: var < end
            point.parallel = rowParallel;
            point.vectorHint = rowVector;
            dimAtStack.push_back(static_cast<int32_t>(varDim));
            const SmallVector<uint32_t, 8> saved = body;
            body.clear();
            MLK_TRYV(emitLevel(b.segStmts[si], r + 1));
            point.children.clear();
            for (const uint32_t c : body) point.children.push_back(c);
            body = saved;
            dimAtStack.pop_back();
            const uint32_t pointId = out.addNode(point);
            out.nodes[tileId].children.push_back(pointId);
        }
        dimAtStack.pop_back();  // tile slot
        body.push_back(tileId);
        return {};
    }

    /// Builds the singleton-tile point segments for every group value
    /// inside [pLo, pHi] and appends the resulting SplitBody.
    void buildTileSegBody(SmallVector<SplitBody, 8>& bodies,
                          const SmallVector<
                              std::pair<int64_t,
                                        SmallVector<uint32_t, 4>>, 4>&
                              groups,
                          const SmallVector<uint32_t, 8>& varStmts,
                          const int64_t tv, const int64_t tileSize,
                          const int64_t pHi) {
        SplitBody b;
        b.tiled = true;
        b.lo = tv;
        b.hi = tv;
        const int64_t pLo = tv * tileSize;
        int64_t cur = pLo;
        for (const auto& g : groups) {
            if (g.first / tileSize != tv) continue;
            if (g.first > cur) {
                b.segRange.emplace_back(cur, g.first - 1);
                b.segStmts.push_back(varStmts);
            }
            SmallVector<uint32_t, 8> segStmts = varStmts;
            for (const uint32_t s : g.second) segStmts.push_back(s);
            b.segRange.emplace_back(g.first, g.first);
            b.segStmts.push_back(std::move(segStmts));
            cur = g.first + 1;
        }
        if (cur <= pHi) {
            b.segRange.emplace_back(cur, pHi);
            b.segStmts.push_back(varStmts);
        }
        bodies.push_back(std::move(b));
    }

    /// Piecewise split emission for a level with re-entry statements.
    /// Falls back to the guarded form when the replay budget would be
    /// exceeded (the guard path is always available and always correct).
    [[nodiscard]] Result<void> emitSplitNest(
        const SmallVector<uint32_t, 8>& varStmts, uint32_t pivotDim,
        int64_t lo, int64_t hi, uint32_t r,
        const SmallVector<std::pair<int64_t, uint32_t>, 8>& reentry) {
        // Group the re-entry statements by folded row value (ascending;
        // deterministic — the classification order is a registry order).
        SmallVector<int64_t, 4> sorted;
        for (const auto& item : reentry) sorted.push_back(item.first);
        std::sort(sorted.begin(), sorted.end());
        SmallVector<int64_t, 4> values;
        for (const int64_t v : sorted) {
            if (values.empty() || values.back() != v) values.push_back(v);
        }
        SmallVector<std::pair<int64_t, SmallVector<uint32_t, 4>>, 4>
            groups;
        for (const int64_t v : values) {
            SmallVector<uint32_t, 4> stmts;
            for (const auto& item : reentry) {
                if (item.first == v) stmts.push_back(item.second);
            }
            groups.emplace_back(v, std::move(stmts));
        }
        const bool tiledHere = tiled->tiled && r < tiled->bandEnd;
        const int64_t tileSize = tiledHere ? tiled->tileSizes[r] : 0;
        // Build the replayed-body list.
        SmallVector<SplitBody, 8> bodies;
        if (!tiledHere) {
            int64_t cur = lo;
            for (const auto& g : groups) {
                if (g.first > cur) {
                    SplitBody b;
                    b.lo = cur;
                    b.hi = g.first - 1;
                    b.stmts = varStmts;
                    bodies.push_back(std::move(b));
                }
                SplitBody b;
                b.lo = g.first;
                b.hi = g.first;
                b.stmts = varStmts;
                for (const uint32_t s : g.second) b.stmts.push_back(s);
                bodies.push_back(std::move(b));
                cur = g.first + 1;
            }
            if (cur <= hi) {
                SplitBody b;
                b.lo = cur;
                b.hi = hi;
                b.stmts = varStmts;
                bodies.push_back(std::move(b));
            }
        } else {
            if (lo % tileSize != 0) {
                fail("codegen: tile lower bound not aligned");
                return {};
            }
            const int64_t firstTile = lo / tileSize;
            const int64_t lastTile = hi / tileSize;
            const int64_t fullLast = (hi + 1) / tileSize - 1;
            const bool hasFull = fullLast >= firstTile;
            const bool hasPartial = (hi + 1) % tileSize != 0;
            // One entry per tile part (full part + trailing partial
            // tile), each cut at the tile indices holding a guard value.
            const std::pair<int64_t, int64_t> parts[] = {
                {firstTile, fullLast}, {lastTile, lastTile}};
            for (int pi = 0; pi < 2; ++pi) {
                const int64_t a = parts[pi].first;
                const int64_t b = parts[pi].second;
                const bool isPartial = pi == 1 && hasPartial;
                if (pi == 1 && !hasPartial) continue;
                if (pi == 0 && !hasFull) continue;
                SmallVector<int64_t, 4> tvs;
                for (const auto& g : groups) {
                    const int64_t tv = g.first / tileSize;
                    if (tv >= a && tv <= b &&
                        (tvs.empty() || tvs.back() != tv)) {
                        tvs.push_back(tv);
                    }
                }
                if (tvs.empty()) {
                    SplitBody sb;
                    sb.tiled = true;
                    sb.lo = a;
                    sb.hi = b;
                    sb.partialEnd = isPartial ? hi : -1;
                    sb.stmts = varStmts;
                    bodies.push_back(std::move(sb));
                    continue;
                }
                int64_t cur = a;
                for (const int64_t tv : tvs) {
                    if (tv > cur) {
                        SplitBody sb;
                        sb.tiled = true;
                        sb.lo = cur;
                        sb.hi = tv - 1;
                        sb.stmts = varStmts;
                        bodies.push_back(std::move(sb));
                    }
                    buildTileSegBody(bodies, groups, varStmts, tv,
                                     tileSize,
                                     isPartial && tv == b
                                         ? hi
                                         : tv * tileSize + tileSize - 1);
                    cur = tv + 1;
                }
                if (cur <= b) {
                    SplitBody sb;
                    sb.tiled = true;
                    sb.lo = cur;
                    sb.hi = b;
                    sb.partialEnd = isPartial ? hi : -1;
                    sb.stmts = varStmts;
                    bodies.push_back(std::move(sb));
                }
            }
        }
        // Budget: every body replays the deeper recursion.
        const int64_t replays = static_cast<int64_t>(bodies.size());
        if (replays > 1 &&
            copyBudget * replays > constants::kPolyMaxCodegenCopies) {
            // Guarded fallback (the CLAST-lite form — correct, retains
            // the per-iteration branch).
            for (const auto& item : reentry) {
                pendingGuards[item.second].push_back(
                    {pivotDim, item.first});
            }
            SmallVector<uint32_t, 8> loopStmts = varStmts;
            for (const auto& item : reentry) {
                loopStmts.push_back(item.second);
            }
            MLK_TRYV(emitLoop(std::move(loopStmts), pivotDim, lo, hi, r));
            return {};
        }
        copyBudget *= replays;
        const SmallVector<bool, 8> snapE = emitted;
        const SmallVector<SmallVector<std::pair<uint32_t, int64_t>, 4>, 8>
            snapG = pendingGuards;
        for (std::size_t bi = 0; bi < bodies.size(); ++bi) {
            if (bi > 0) {
                // Restore replay marks ONLY for the statements this
                // body replays (its transitive emission set is exactly
                // its own statement lists — deeper re-entries are drawn
                // from the lists passed down). Payloads emitted in
                // earlier bodies alone keep their marks. Guards restore
                // wholesale (deeper fallback levels re-push them).
                for (const uint32_t s : bodies[bi].stmts) {
                    emitted[s] = snapE[s];
                }
                for (const auto& seg : bodies[bi].segStmts) {
                    for (const uint32_t s : seg) emitted[s] = snapE[s];
                }
                pendingGuards = snapG;
            }
            MLK_TRYV(emitSplitBody(bodies[bi], pivotDim, r));
        }
        return {};
    }

    Result<void> emitLevel(SmallVector<uint32_t, 8> stmts, uint32_t r) {
        if (failed) return {};
        SmallVector<uint32_t, 8> alive;
        for (const uint32_t s : stmts) {
            if (!emitted[s]) alive.push_back(s);
        }
        if (alive.empty()) return {};
        if (r >= sched->rows.size() || r >= sched->pivotDim.size()) {
            // Schedule exhausted: the scheduler's totality contract means
            // remaining statements have no unspent varying dims (each is
            // emitted exactly once). The recursion list order is the
            // classification order (varying first, guarded re-entries
            // appended) — reorder by origOrder so dependence-tied
            // statement pairs replay in program order (Rule 53).
            SmallVector<uint32_t, 8> ordered = alive;
            std::sort(ordered.begin(), ordered.end(),
                      [&](uint32_t a, uint32_t b) {
                          return scop->statements[a].origOrder <
                                 scop->statements[b].origOrder;
                      });
            for (const uint32_t s : ordered) {
                MLK_TRYV(emitPayload(scop->statements[s]));
                emitted[s] = true;
            }
            return {};
        }
        const uint32_t pivot = sched->pivotDim[r];
        const ScheduleRow& row = sched->rows[r];
        // Classify alive statements at this level: foldable constants vs
        // pivot-varying (the scheduler's shape contract admits exactly
        // these two classes; anything else is a contract bug).
        SmallVector<std::pair<int64_t, uint32_t>, 8> constItems;
        SmallVector<uint32_t, 8> varStmts;
        for (const uint32_t s : alive) {
            const Statement& st = scop->statements[s];
            int64_t folded = 0;
            if (detail::rowEffectiveConst(st, row, &folded)) {
                constItems.emplace_back(folded, s);
                continue;
            }
            const auto& coeffs = row.stmtCoeffs[st.id];
            if (pivot + 1 < coeffs.size() && coeffs[pivot + 1] >= 1) {
                varStmts.push_back(s);
                continue;
            }
            fail("codegen: statement neither constant nor pivot-varying");
            return {};
        }
        if (varStmts.empty()) {
            MLK_TRYV(emitConstItems(std::move(constItems)));
            return {};
        }
        // Loop bounds: box of the pivot dim, intersected over varStmts.
        // Skew coefficients on spent dims are order-only and never reach
        // bounds (the nest order equals the schedule lex order).
        int64_t lo = INT64_MIN;
        int64_t hi = INT64_MAX;
        for (const uint32_t s : varStmts) {
            const Statement& st = scop->statements[s];
            if (pivot >= st.ownLower.size() || pivot >= st.ownUpper.size()) {
                fail("codegen: statement lacks bounds for the loop dim");
                return {};
            }
            lo = st.ownLower[pivot] > lo ? st.ownLower[pivot] : lo;
            hi = st.ownUpper[pivot] < hi ? st.ownUpper[pivot] : hi;
        }
        if (lo == INT64_MIN || hi == INT64_MAX || lo > hi) {
            fail("codegen: empty or unbounded loop range");
            return {};
        }
        // Constant statements around the loop range.
        SmallVector<std::pair<int64_t, uint32_t>, 8> before, after;
        SmallVector<std::pair<int64_t, uint32_t>, 8> reentry;
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
                // CLAST re-entry: the statement's row value sits inside
                // the loop range (or hoisting is unsafe), so its
                // remaining schedule rows continue INSIDE the loop at
                // the pivot coordinate "value". Realized by the
                // PIECEWISE SPLIT (emitSplitNest): the loop range is
                // cut at every re-entry value and the statement joins
                // only the singleton segment [v, v], where the guard
                // condition "pivot var == value" holds structurally —
                // the hot segments lose the branch entirely. The
                // per-iteration Guard form remains as the fallback when
                // the split's replay budget would be exceeded.
                // Soundness: the scheduler's shape contract pins the
                // pivot for every row-constant statement (its instances
                // occupy exactly that coordinate) and the realizability
                // gate keeps the pivot-coordinate order forward on every
                // live pair, so the segment (or guard) fires at the
                // statement's schedule position and the fused deeper
                // loops replay it exactly once. Segment enumeration in
                // ascending pivot order preserves the guarded
                // execution order bit-exactly.
                reentry.push_back(item);
            }
        }
        MLK_TRYV(emitConstItems(std::move(before)));
        if (reentry.empty()) {
            SmallVector<uint32_t, 8> nestStmts = varStmts;
            MLK_TRYV(emitLoop(std::move(nestStmts), pivot, lo, hi, r));
        } else {
            // Piecewise split: realize every re-entry guard
            // structurally (singleton point segments — the guard
            // condition holds for the WHOLE segment); the hot segments
            // lose the branch. Falls back to the guarded form above
            // when the replay budget would be exceeded.
            SmallVector<std::pair<int64_t, uint32_t>, 8> reentryItems =
                reentry;
            MLK_TRYV(emitSplitNest(varStmts, pivot, lo, hi, r,
                                   reentryItems));
        }
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
    em.pendingGuards = SmallVector<
        SmallVector<std::pair<uint32_t, int64_t>, 4>, 8>(
        scop.statements.size());
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
