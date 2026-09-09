# Tier 0 reference interpreter backend (interface target over mlk_runtime's
# reference interpreter — single implementation, no duplication: Rule 77).
add_library(mlk_backend_interpreter INTERFACE)
target_link_libraries(mlk_backend_interpreter INTERFACE mlk_runtime)
