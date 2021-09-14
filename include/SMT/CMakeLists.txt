add_mlir_dialect(SMTOps smt)
add_mlir_doc(SMTDialect SMTDialect SMT/ -gen-dialect-doc)
add_mlir_doc(SMTOps SMTOps SMT/ -gen-op-doc)

add_mlir_interface(SMTInterfaces)
add_dependencies(MLIRSMTOpsIncGen MLIRSMTInterfacesIncGen)
