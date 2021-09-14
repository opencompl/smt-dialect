// Translation from MLIR (SMT dialect) to SMTLib code.

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Translation.h"

#include "SMT/SMTOps.h"

using namespace mlir;
using namespace smt;

namespace {

class SMTTranslation {
  FuncOp getSMTMain(ModuleOp module) {
    auto funcOps = module.getRegion().getOps<FuncOp>();
    FuncOp mainFunc = nullptr;
    for (auto func : funcOps) {
      if (func->hasAttr("smt_main")) {
        if (mainFunc) {
          func->emitWarning("Ignoring redefinition of `smt_main` function " +
                            func.sym_name());
        } else {
          mainFunc = func;
        }
      }
    }
    if (!mainFunc) {
      module->emitError(
          "Missing main function - no function with attribute `smt_main`");
    }
    return mainFunc;
  }

public:
  SMTTranslation(raw_ostream &output) : output(output) {}
  LogicalResult translateToSMT(ModuleOp module) {
    FuncOp mainFunc;
    if (!(mainFunc = getSMTMain(module)))
      return failure();
    output << ";;; Translating MLIR to SMTLib code...\n";
    module.walk([&](Operation *op) {
      if (auto translateInterface = dyn_cast<SMTTranslateOpInterface>(op)) {
        translateInterface.translateToSMT(output);
      }
    });
    return success();
  }

private:
  raw_ostream &output;
};
} // namespace

namespace mlir {
void registerMLIRToSMTTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-smt",
      [](ModuleOp module, raw_ostream &output) {
        SMTTranslation translation(output);
        return translation.translateToSMT(module);
      },
      [](DialectRegistry &registry) {
        registry.insert<StandardOpsDialect, smt::SMTDialect>();
      });
}
} // namespace mlir
