add_mlir_translation_library(MLIRToSMTTranslation
  MLIRToSMT.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSMT
)
