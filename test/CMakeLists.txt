configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SMT_OPT_TEST_DEPENDS
        FileCheck count not
        smt-opt
        )

add_lit_testsuite(check-smt-opt "Running the smt-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${SMT_OPT_TEST_DEPENDS}
        )
set_target_properties(check-smt-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(SMT_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${SMT_OPT_TEST_DEPENDS})
