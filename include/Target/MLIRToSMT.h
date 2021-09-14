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

  // Add a (define-fun), if the symbol does not already exist.
  LogicalResult addFuncDef(FlatSymbolRefAttr);

  // Serialize an SMT dialect op into an SMT statement
  LogicalResult serializeStatement(Operation *op, std::string &expr);

  // Serialize a `value` into an SMT expression and append to `expr`
  LogicalResult serializeExpression(Value value, std::string &expr);
};

} // namespace smt
} // namespace mlir

#endif // TARGET_MLIRTOSMT_H_