// Translation from MLIR (SMT dialect) to SMTLib code.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "SMT/SMTOps.h"
#include "SMT/SMTSerializers.h"
#include "Target/MLIRToSMT.h"

using namespace mlir;
using namespace smt;

namespace {

//===== MLIR To SMT Translation Module =======================================//
class SMTTranslation {
  func::FuncOp getSMTMain(ModuleOp module) {
    auto funcOps = module.getRegion().getOps<func::FuncOp>();
    func::FuncOp mainFunc = nullptr;
    for (auto func : funcOps) {
      if (func->hasAttr("smt_main")) {
        if (mainFunc) {
          func->emitWarning(
              "[mlir-to-smt] Ignoring redefinition of `smt_main` function " +
              func.getSymName());
        } else {
          mainFunc = func;
        }
      }
    }
    if (!mainFunc) {
      module->emitError("[mlir-to-smt] Missing main function - no function "
                        "with attribute `smt_main`");
    }
    return mainFunc;
  }

public:
  SMTTranslation(raw_ostream &output) : output(output) {}
  LogicalResult translateToSMT(ModuleOp module) {
    func::FuncOp mainFunc;
    if (!(mainFunc = getSMTMain(module)))
      return failure();
    SMTContext smtContext(module, output, *module->getContext());
    auto walkResult = module.walk([&](Operation *op) {
      std::string expr;
      llvm::raw_string_ostream os(expr);
      if (failed(smtContext.serializeStatement(op, os))) {
        return WalkResult::interrupt();
      }
      if (!expr.empty())
        output << expr << '\n';
      return WalkResult::advance();
    });
    return success(!walkResult.wasInterrupted());
  }

private:
  raw_ostream &output;
};
} // namespace

namespace mlir {
void registerMLIRToSMTTranslation(
    std::function<void(DialectRegistry &)> registrationCallback) {
  TranslateFromMLIRRegistration registration(
      "mlir-to-smt", "Translate to an SMTLib script",
      [](ModuleOp module, raw_ostream &output) {
        SMTTranslation translation(output);
        return translation.translateToSMT(module);
      },
      [&](DialectRegistry &registry) {
        registry.insert<func::FuncDialect, smt::SMTDialect>();
        registrationCallback(registry);
      });
}
} // namespace mlir
