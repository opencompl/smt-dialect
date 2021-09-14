// Translation from MLIR (SMT dialect) to SMTLib code.

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Translation.h"

#include "SMT/SMTOps.h"
#include "Target/MLIRToSMT.h"

using namespace mlir;
using namespace smt;

namespace {

//===== SMT Operation Converters =============================================//
LogicalResult serializeSMTStatement(CheckSatOp op, std::string &expr,
                                    SMTContext &ctx) {
  expr = "(check-sat)";
  return success();
}
LogicalResult serializeSMTStatement(GetModelOp op, std::string &expr,
                                    SMTContext &ctx) {
  expr = "(get-model)";
  return success();
}
LogicalResult serializeSMTStatement(smt::AssertOp op, std::string &expr,
                                    SMTContext &ctx) {
  expr = "(assert ";
  if (failed(ctx.serializeExpression(op.cond(), expr)))
    return failure();
  expr += ")";
  return success();
}

//===== MLIR To SMT Translation Module =======================================//
class SMTTranslation {
  FuncOp getSMTMain(ModuleOp module) {
    auto funcOps = module.getRegion().getOps<FuncOp>();
    FuncOp mainFunc = nullptr;
    for (auto func : funcOps) {
      if (func->hasAttr("smt_main")) {
        if (mainFunc) {
          func->emitWarning(
              "[mlir-to-smt] Ignoring redefinition of `smt_main` function " +
              func.sym_name());
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

  template <typename Op>
  LogicalResult serialize(Operation *op, std::string &expr, SMTContext &ctx) {
    if (auto opp = dyn_cast<Op>(op)) {
      return serializeSMTStatement(opp, expr, ctx);
    }
    return success();
  }
  template <typename Op, typename T, typename... Ts>
  LogicalResult serialize(Operation *op, std::string &expr, SMTContext &ctx) {
    if (auto opp = dyn_cast<Op>(op)) {
      return serializeSMTStatement(opp, expr, ctx);
    }
    return serialize<T, Ts...>(op, expr, ctx);
  }

public:
  SMTTranslation(raw_ostream &output) : output(output) {}
  LogicalResult translateToSMT(ModuleOp module) {
    FuncOp mainFunc;
    if (!(mainFunc = getSMTMain(module)))
      return failure();
    SMTContext smtContext(module, *module->getContext());
    auto walkResult = module.walk([&](Operation *op) {
      std::string expr;
      if (failed(serialize<CheckSatOp, GetModelOp, smt::AssertOp>(
              op, expr, smtContext))) {
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
