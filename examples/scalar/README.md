# Scalar example: f(x) = sin(x^2 + 3x)

The spec's worked example, as a .mlk graph.

```
# inspect the canonical IR (tier 1 pipeline)
mlkc compile sin_graph.mlk --tier=1 --emit=ir

# evaluate at x = 2 (expect -0.5440211108893698)
mlkc run sin_graph.mlk --tier=0 --x2.0

# all tiers must agree (Rule 43 differential law)
for t in 0 1 2 3; do mlkc run sin_graph.mlk --tier=$t --x2.0; done

# verify + tune
mlk-verify sin_graph.mlk
mlk-tune tune sin_graph.mlk

# emit C++ kernel source (AOT path, spec section 12)
mlkc compile sin_graph.mlk --tier=1 --emit=cpp
```
