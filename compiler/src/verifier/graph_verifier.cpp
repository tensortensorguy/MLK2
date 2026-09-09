// Graph verifier implementation (Rules 47, 145).
#include "mlk/verifier/graph_verifier.h"

#include "mlk/effect/effect_inference.h"
#include "mlk/ir/graph_printer.h"
#include "mlk/type/type_inference.h"

namespace mlk {

bool verifyGraph(const MathGraph& graph, const MathDomainProfile* profile,
                 VerifyOptions opts, DiagnosticEngine& diag) {
    bool ok = true;
    const auto fail = [&](NodeId node, std::string what, std::string expected,
                          std::string actual, std::string rule,
                          std::string fix) {
        diag.reportError(node, std::move(what), std::move(expected),
                         std::move(actual), std::move(rule), std::move(fix));
        ok = false;
    };

    // 1. Index integrity: no dangling NodeId/ValueId (Rule 47).
    for (const auto& n : graph.nodes()) {
        if (n.flags.test(NodeFlag::Dead)) continue;
        for (const ValueId in : n.inputs) {
            if (in >= graph.numValues()) {
                fail(n.id, "dangling operand reference",
                     "operand id < numValues",
                     "operand id " + std::to_string(in) + " >= " +
                         std::to_string(graph.numValues()),
                     "Rule 47", "rebuild the graph with graph.rebuildUsers()");
            }
        }
        for (const ValueId out : n.results) {
            if (out >= graph.numValues()) {
                fail(n.id, "dangling result reference",
                     "result id < numValues",
                     "result id " + std::to_string(out), "Rule 47",
                     "recreate the node");
            }
        }
        // 2. Arity legality.
        if (!arityAccepts(n.op, n.numInputs())) {
            fail(n.id, std::string("illegal arity for op ") + opName(n.op),
                 std::to_string(opInfo(n.op).arity.min) + ".." +
                     std::to_string(opInfo(n.op).arity.max),
                 std::to_string(n.numInputs()), "Rule 47",
                 "check the op table in math_op.h");
        }
    }

    // 3. Outputs defined.
    for (const ValueId out : graph.outputs()) {
        if (out >= graph.numValues()) {
            fail(kInvalidNodeId, "output id out of range",
                 "output id < numValues", std::to_string(out), "Rule 47",
                 "fix graph outputs");
            continue;
        }
        const Value& v = graph.value(out);
        if (v.kind == ValueKind::NodeResult &&
            graph.isNodeDead(v.producer)) {
            fail(v.producer, "output produced by dead node",
                 "output producer alive", "producer killed", "Rule 92",
                 "re-derive the output or remove it from outputs()");
        }
    }

    // 4. Use-def consistency.
    for (const auto& v : graph.values()) {
        if (v.kind == ValueKind::NodeResult &&
            v.producer >= graph.numNodes()) {
            fail(kInvalidNodeId, "node result without producer",
                 "producer < numNodes", std::to_string(v.producer),
                 "Rule 47", "recreate the value");
        }
    }

    // 5. Acyclicity where illegal (Rule 47): node id order is topological
    // by construction; a cycle would mean an operand id >= node id.
    if (opts.checkAcyclic) {
        for (const auto& n : graph.nodes()) {
            if (n.flags.test(NodeFlag::Dead)) continue;
            for (const ValueId in : n.inputs) {
                if (in < graph.numValues()) {
                    const Value& v = graph.value(in);
                    if (v.kind == ValueKind::NodeResult &&
                        v.producer >= n.id) {
                        fail(n.id, "operand produced at or after consumer",
                             "producer id < consumer id",
                             "producer " + std::to_string(v.producer),
                             "Rule 47",
                             "insert a copy or restructure the graph");
                    }
                }
            }
        }
    }

    // 6. Type re-validation (derived types match re-inference).
    if (opts.checkTypes && profile != nullptr) {
        for (const NodeId nid : graph.topoOrder()) {
            const Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || n.inputs.empty()) continue;
            const auto inferred = inferResultType(graph, nid, profile);
            const Value& result = graph.value(n.results[0]);
            // Values whose type has not been inferred yet (domain Unknown)
            // are skipped: type.infer owns first assignment; re-validation
            // applies to derived types only.
            if (result.type.domain == Domain::Unknown) continue;
            if (!inferred.has_value() ||
                !(result.type == *inferred)) {
                fail(nid,
                     std::string("derived type mismatch at ") + opName(n.op),
                     inferred.has_value() ? domainName(inferred->domain)
                                          : "inference failure",
                     domainName(result.type.domain), "Rule 146",
                     "re-run type.infer");
            }
        }
    }

    // 7. Effect chain continuity (Rule 145).
    if (opts.checkEffects) {
        const EffectChain& chain = graph.effectChain();
        for (const NodeId nid : chain.order()) {
            if (nid >= graph.numNodes()) {
                fail(nid, "effect chain references missing node",
                     "chain entry < numNodes", std::to_string(nid),
                     "Rule 145", "splice the effect chain on removal");
                continue;
            }
            const Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead)) {
                fail(nid, "effect chain contains dead node",
                     "chain alive-only", "dead node in chain", "Rule 145",
                     "killNode() splices the chain; use it");
            }
            if (isPure(n.effects)) {
                fail(nid, "pure node in effect chain",
                     "effecting nodes only", "pure node", "Rule 140",
                     "attach effects via addNode()");
            }
        }
    }

    // 8. Guards carry GraphState; speculative nodes carry full metadata
    // (Rules 5, 65, 141, 145).
    if (opts.checkGuards) {
        for (const auto& n : graph.nodes()) {
            if (n.flags.test(NodeFlag::Dead)) continue;
            if (!n.flags.test(NodeFlag::Speculative)) continue;
            if (!n.flags.test(NodeFlag::HasGraphState)) {
                fail(n.id, "speculative node without GraphState",
                     "GraphState attachment present", "missing", "Rule 5",
                     "attach GraphState before marking speculative");
            }
            const SpeculationInfo& si = n.speculation;
            if (si.source == SpeculationSource::None) {
                fail(n.id, "speculation without evidence source",
                     "Profile|Benchmark|StaticProof", "None", "Rule 141",
                     "record the evidence source");
            }
            if (si.fallbackTarget == kInvalidSymbolId) {
                fail(n.id, "speculation without fallback target",
                     "named fallback target", "invalid symbol", "Rule 62",
                     "declare the generic fallback");
            }
        }
    }

    // 9. Property lattice consistency: no contradictory definite facts
    // (Rule 22/47).
    if (opts.checkFacts) {
        for (const auto& v : graph.values()) {
            const Fact* pos = v.facts.get(PropertyId::MonotonicIncreasing);
            const Fact* neg = v.facts.get(PropertyId::MonotonicDecreasing);
            if (pos != nullptr && neg != nullptr &&
                pos->value == TriState::True && neg->value == TriState::True) {
                fail(v.producer,
                     "contradictory monotonicity facts on one value",
                     "at most one definite direction", "both True",
                     "Rule 22", "re-run property.infer");
            }
        }
    }

    return ok;
}

}  // namespace mlk
