#ifndef SMT_SMTSERIALIZERS_H_
#define SMT_SMTSERIALIZERS_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

#include <map>

namespace mlir {
namespace smt {

class SMTContext {
  ModuleOp module;
  llvm::raw_ostream &primaryOS;
  MLIRContext &ctx;
  // function declarations (uninterpreted) and definitions
  std::map<std::string, std::string> funcDefs;

public:
  SMTContext(ModuleOp module, llvm::raw_ostream &primaryOS, MLIRContext &ctx)
      : module(module), primaryOS(primaryOS), ctx(ctx), funcDefs() {}

  // Add a function definition, if the symbol does not already exist.
  // If the body is empty, add an uninterpreted (declare-fun)
  // Otherwise generate a (define-fun)
  LogicalResult addFunc(StringRef funcName, FunctionType funcType, std::string funcBody = "");

  // Convert an existing function in the module
  LogicalResult addMLIRFunction(StringRef funcName);

  // Print a generic type
  LogicalResult printGenericType(Type type, llvm::raw_ostream &os);

  // Serialize an SMT dialect op into an SMT statement
  LogicalResult serializeStatement(Operation *op, llvm::raw_ostream &os);

  // Serialize a list of values, space separated
  LogicalResult serializeArguments(ValueRange valueRange,
                                   llvm::raw_ostream &expr);

  // Serialize a `value` into an SMT expression and append to `expr`
  LogicalResult serializeExpression(Value value, llvm::raw_ostream &expr);
};

} // namespace smt
} // namespace mlir

#endif // SMT_SMTSERIALIZERS_H_