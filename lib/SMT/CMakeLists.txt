add_mlir_dialect_library(MLIRSMT
  SMTDialect.cpp
  SMTOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/SMT

  DEPENDS
  MLIRSMTOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
