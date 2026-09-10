// PolyWorkspace lifecycle (see workspace.h).
#include "mlk/poly/workspace.h"

namespace mlk::poly {

PolyWorkspace* createPolyWorkspace() { return new PolyWorkspace{}; }

void destroyPolyWorkspace(PolyWorkspace* ws) noexcept { delete ws; }

}  // namespace mlk::poly
