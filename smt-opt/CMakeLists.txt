get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRSMT
        )
add_llvm_executable(smt-opt smt-opt.cpp)

llvm_update_compile_flags(smt-opt)
target_link_libraries(smt-opt PRIVATE ${LIBS})
