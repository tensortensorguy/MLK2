# Unit tests (Rule 80): one executable per suite for isolation.
foreach(suite IN ITEMS
        unit_core unit_ir unit_passes_math unit_passes_analysis
        unit_egraph unit_calculus unit_tensor unit_runtime unit_autotune
        unit_superopt unit_golden unit_poly unit_fastkernel)
    mlk_add_test(mlk_${suite}
        SOURCES ${suite}.cpp
        DEPS mlk_runtime mlk_autotune mlk_superopt mlk_backend_cpu
            mlk_fastkernel
    )
endforeach()
