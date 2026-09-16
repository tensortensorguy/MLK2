# mlk_fastkernel — certified, budgeted fast-kernel search
# (OEIA/Refined/FastKernels; docs/polyhedral_spec.md §fast-kernel-search).
#
# Dependency flow (strict, one-directional, Rule 70):
#   mlk_pipeline -> mlk_backend_cpu -> mlk_fastkernel
# The search orchestrates the pipeline (per-variant Tier2 compiles), the
# out-of-process artifact driver, and the buffer executor; it owns no IR
# of its own.
mlk_add_library(mlk_fastkernel
    INCLUDE_DIR include
    SOURCES
        src/fast_kernel.cpp
    PUBLIC_DEPS mlk_backend_cpu mlk_pipeline
)

set(MLK_FASTKERNEL_LIB mlk_fastkernel PARENT_SCOPE)
