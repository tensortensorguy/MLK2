# Tool integrations: clang-format / clang-tidy targets (Rule 66, scripts/).

find_program(CLANG_FORMAT_EXE NAMES clang-format clang-format-18 clang-format-17 clang-format-16)
find_program(CLANG_TIDY_EXE NAMES clang-tidy clang-tidy-18 clang-tidy-17)

if(CLANG_FORMAT_EXE)
    add_custom_target(mlk-format
        COMMAND ${CLANG_FORMAT_EXE} -i
            $<JOIN:${MLK_ALL_SOURCES},>
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "clang-format all MLK+ sources")
endif()

if(CLANG_TIDY_EXE)
    add_custom_target(mlk-tidy
        COMMAND ${CLANG_TIDY_EXE} -p ${CMAKE_BINARY_DIR} ${MLK_ALL_SOURCES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "clang-tidy MLK+ sources")
endif()

# Slop lint (Rule 84): banned patterns in hot IR code. Run via scripts/lint.sh.
add_custom_target(mlk-lint
    COMMAND /bin/bash ${CMAKE_SOURCE_DIR}/scripts/lint.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "MLK+ anti-slop lint (Rule 84 checklist)")
