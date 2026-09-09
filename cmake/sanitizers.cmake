# Sanitizer configuration (Rule 153: sanitizer matrix is mandatory).
#
# Usage:
#   cmake -DMLK_SANITIZER=address|undefined|thread|leak ..

set(MLK_SANITIZER "" CACHE STRING "Sanitizer: address|undefined|thread|leak")

function(mlk_apply_sanitizers target)
    if(MLK_SANITIZER STREQUAL "address")
        if(MSVC)
            target_compile_options(${target} PRIVATE /fsanitize=address)
        else()
            target_compile_options(${target} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
            target_link_options(${target} PRIVATE -fsanitize=address)
        endif()
    elseif(MLK_SANITIZER STREQUAL "undefined")
        if(NOT MSVC)
            # Rule 6 note: -fno-sanitize=vptr because RTTI is disabled (Rule 8).
            target_compile_options(${target} PRIVATE -fsanitize=undefined,implicit-conversion -fno-sanitize=vptr)
            target_link_options(${target} PRIVATE -fsanitize=undefined)
        endif()
    elseif(MLK_SANITIZER STREQUAL "thread")
        if(NOT MSVC)
            target_compile_options(${target} PRIVATE -fsanitize=thread)
            target_link_options(${target} PRIVATE -fsanitize=thread)
        endif()
    elseif(MLK_SANITIZER STREQUAL "leak")
        if(NOT MSVC)
            target_compile_options(${target} PRIVATE -fsanitize=leak)
            target_link_options(${target} PRIVATE -fsanitize=leak)
        endif()
    elseif(NOT MLK_SANITIZER STREQUAL "")
        message(FATAL_ERROR "Unknown MLK_SANITIZER '${MLK_SANITIZER}'")
    endif()
endfunction()
