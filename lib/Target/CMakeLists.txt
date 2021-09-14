add_mlir_translation_library(MLIRToSMTTranslation
  MLIRToSMT.cpp
  Serializers.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Target

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSMT
)
