#ifndef TARGET_MLIRTOSMT_H_
#define TARGET_MLIRTOSMT_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

#include <map>

namespace mlir {
namespace smt {

class SMTContext {
  ModuleOp module;
  MLIRContext &ctx;
  std::map<Value, std::string> funcDefs;

public:
  SMTContext(ModuleOp module, MLIRContext &ctx)
      : module(module), ctx(ctx), funcDefs() {}
  LogicalResult addFuncDef(FlatSymbolRefAttr);
};

} // namespace smt
} // namespace mlir

#endif // TARGET_MLIRTOSMT_H_