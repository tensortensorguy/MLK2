// MLK+ polyhedral engine — umbrella header.
//
// The engine is the hermetic Presburger/affine toolkit behind the poly.*
// pass family: exact rational arithmetic, integer affine expressions,
// disjunctive Presburger sets (FM elimination, lexmin/lexmax), and affine
// maps (image/preimage). No external dependencies (Rule 160), no
// exceptions (Rule 6), no RTTI (Rule 8).
#pragma once

#include "mlk/poly/affine_expr.h"
#include "mlk/poly/affine_map.h"
#include "mlk/poly/dependence.h"
#include "mlk/poly/int_set.h"
#include "mlk/poly/lp.h"
#include "mlk/poly/pluto.h"
#include "mlk/poly/rational.h"
#include "mlk/poly/scop.h"
#include "mlk/poly/workspace.h"
