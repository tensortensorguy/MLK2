// MLK+ graph builder (frontend-facing fluent construction API).
#pragma once

#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/math_graph.h"
#include "mlk/type/math_type.h"

namespace mlk {

/// Fluent builder over MathGraph for frontends, tests, and examples.
class GraphBuilder {
public:
    explicit GraphBuilder(SymbolTable& symbols) : symbols_(symbols) {}

    [[nodiscard]] SymbolTable& symbols() noexcept { return symbols_; }

    [[nodiscard]] ValueId placeholder(const char* name, MathType type) {
        return graph_.addPlaceholder(symbols_.intern(name), std::move(type));
    }
    [[nodiscard]] ValueId variable(const char* name, MathType type) {
        return graph_.addVariable(symbols_.intern(name), std::move(type));
    }
    [[nodiscard]] ValueId constant(double v, MathType type) {
        return graph_.addConstant(v, std::move(type));
    }
    [[nodiscard]] ValueId integer(int64_t v, MathType type) {
        return graph_.addIntConstant(v, std::move(type));
    }
    [[nodiscard]] ValueId symbol(const char* name, MathType type) {
        return graph_.addSymbol(symbols_.intern(name), std::move(type));
    }

    [[nodiscard]] Result<ValueId> op(MathOp op,
                                     std::initializer_list<ValueId> inputs,
                                     AttrList attrs = {}) {
        SmallVector<ValueId, 4> in;
        for (const ValueId v : inputs) in.push_back(v);
        MLK_TRY_VAR(vid, graph_.addNode(op, in, std::move(attrs)));
        return vid;
    }

    MathGraph& graph() noexcept { return graph_; }
    [[nodiscard]] const MathGraph& graph() const noexcept { return graph_; }

    void output(ValueId v) { graph_.addOutput(v); }

private:
    SymbolTable& symbols_;
    MathGraph graph_{&symbols_};
};

}  // namespace mlk
