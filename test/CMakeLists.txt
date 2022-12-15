if (MAIN_PROJECT)
configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CHECK_SMT_TEST_DEPENDS
        FileCheck count not
        smt-opt
        smt-translate
        )

add_lit_testsuite(check-smt "Running the smt-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${CHECK_SMT_TEST_DEPENDS}
        )
set_target_properties(check-smt PROPERTIES FOLDER "Tests")

add_lit_testsuites(CHECK_SMT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${CHECK_SMT_TEST_DEPENDS})
endif()
