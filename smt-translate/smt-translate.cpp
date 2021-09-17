// SMT2 <-> MLIR Translation tool

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"

#include "SMT/SMTOps.h"
#include "Target/MLIRToSMT.h"

using namespace mlir;

int main(int argc, char **argv) {
  registerMLIRToSMTTranslation();
  registerAllTranslations();
  return failed(mlirTranslateMain(argc, argv, "SMT-MLIR Translation Tool"));
}
