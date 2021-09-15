#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "SMT/SMTOps.h"
#include "Target/MLIRToSMT.h"

using namespace mlir;
using namespace smt;

namespace {

//==== Operation Serializers ================================================//

// Registering a custom serializer:
// *** Use this only if you cannot add the SMTSerializeOpInterface directly to
// the Op ***
//
// Define the following function and register the operation in
// SMTContext::serializeExpression (i.e. add to template list)
// LogicalResult serializeExpression(MyOperation op, std::string &expr,
//                                   SMTContext &ctx);
// * Append the serialized expression to `expr`.

LogicalResult serializeExpression(ConstantIntOp op, std::string &expr,
                                  SMTContext &ctx) {
  unsigned width = op.getType().cast<IntegerType>().getWidth();
  int64_t value = op.getValue();
  if (width == 1) {
    // Boolean value
    expr += value ? "true" : "false";
    return success();
  }
  expr += std::to_string(value);
  return success();
}
LogicalResult serializeExpression(CmpIOp op, std::string &expr,
                                  SMTContext &ctx) {
  std::string args;
  if (failed(ctx.serializeExpression(op.lhs(), args)))
    return failure();
  args += " ";
  if (failed(ctx.serializeExpression(op.rhs(), args)))
    return failure();
  switch (op.getPredicate()) {
  case CmpIPredicate::eq:
    expr += "(= " + args + ")";
    return success();
  case CmpIPredicate::ne:
    expr += "(not (= " + args + "))";
    return success();
  case CmpIPredicate::sge:
  case CmpIPredicate::uge:
    expr += "(>= " + args + ")";
    return success();
  case CmpIPredicate::sgt:
  case CmpIPredicate::ugt:
    expr += "(> " + args + ")";
    return success();
  case CmpIPredicate::sle:
  case CmpIPredicate::ule:
    expr += "(<= " + args + ")";
    return success();
  case CmpIPredicate::slt:
  case CmpIPredicate::ult:
    expr += "(< " + args + ")";
    return success();
  }
}

template <typename Op>
LogicalResult serializeBinOp(const char *opSym, Op op, std::string &expr,
                             SMTContext &ctx) {
  expr += "(" + std::string(opSym) + " ";
  if (failed(ctx.serializeExpression(op.lhs(), expr)))
    return failure();
  expr += " ";
  if (failed(ctx.serializeExpression(op.rhs(), expr)))
    return failure();
  expr += ")";
  return success();
}

LogicalResult serializeExpression(AddIOp op, std::string &expr,
                                  SMTContext &ctx) {
  return serializeBinOp<AddIOp>("+", op, expr, ctx);
}
LogicalResult serializeExpression(MulIOp op, std::string &expr,
                                  SMTContext &ctx) {
  return serializeBinOp<MulIOp>("*", op, expr, ctx);
}

//==== Expression Serializer Execution =======================================//

template <typename Op>
LogicalResult serializeExpression(Operation *op, std::string &expr,
                                  SMTContext &ctx) {
  if (auto opp = dyn_cast<Op>(op)) {
    return serializeExpression(opp, expr, ctx);
  }
  return op->emitError(
      "[mlir-to-smt] cannot convert operation to SMT expression - no "
      "registered serialization");
}

template <typename Op, typename T, typename... Ts>
LogicalResult serializeExpression(Operation *op, std::string &expr,
                                  SMTContext &ctx) {
  if (auto opp = dyn_cast<Op>(op)) {
    return serializeExpression(opp, expr, ctx);
  }
  return serializeExpression<T, Ts...>(op, expr, ctx);
}
} // namespace

//==== SMTContext ===========================================================//
LogicalResult SMTContext::addFuncDef(StringRef funcName,
                                     FunctionType funcType) {
  if (funcDefs.find(funcName.str()) != funcDefs.end())
    return success();
  if (FuncOp func = dyn_cast<FuncOp>(module.lookupSymbol(funcName))) {
    if (func.getType().getNumResults() != 1) {
      return func.emitError("[mlir-to-smt] only functions with a single return "
                            "value are supported.");
    }
    std::string &def = funcDefs[funcName.str()];
    if (func.isDeclaration())
      def = "(declare-fun ";
    else
      def = "(define-fun ";
    def += funcName;

    // argument list
    def += " (";
    for (auto param : llvm::enumerate(func.getType().getInputs())) {
      def += "(arg";
      def += std::to_string(param.index());
      def += " ";
      if (failed(printGenericType(param.value(), def)))
        return failure();
      def += ")";
    }
    def += ") ";

    // return type
    if (failed(printGenericType(func.getType().getResult(0), def)))
      return failure();

    if (!func.isDeclaration()) {
      // generate the body
      def += " ";
      auto &blocks = func.getRegion().getBlocks();
      if (blocks.size() != 1) {
        return func->emitError(
            "[mlir-to-smt] Only functions with a single block supported.");
      }
      if (failed(serializeExpression(
              cast<ReturnOp>(blocks.front().getTerminator()).getOperand(0),
              def)))
        return failure();
    }

    def += ")";

    primaryOS << funcDefs[funcName.str()] << '\n';
    return success();
  }
  return failure();
}

LogicalResult SMTContext::printGenericType(Type type, std::string &expr) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    // TODO: Add support for bitvectors
    if (intType.getWidth() == 1) {
      expr += "Bool";
    } else {
      expr += "Int";
    }
    return success();
  }
  return emitError(UnknownLoc(), "[mlir-to-smt] Unknown type found: ") << type;
}

LogicalResult SMTContext::serializeExpression(Value value, std::string &expr) {
  // Value is block argument, generate the name and return.
  // NOTE: block arguments do not have a defining op.
  if (auto blockArg = value.dyn_cast<BlockArgument>()) {
    expr += "arg" + std::to_string(blockArg.getArgNumber());
    return success();
  }

  Operation *parentOp = value.getDefiningOp();

  // If it has a serializer interface defined, use it directly.
  if (auto serializer = dyn_cast<SMTSerializableOpInterface>(parentOp)) {
    return serializer.serializeExpression(expr, *this);
  }

  // handle function calls separately.
  if (auto callOp = dyn_cast<CallOp>(parentOp)) {
    if (failed(this->addFuncDef(callOp.getCallee(), callOp.getCalleeType())))
      return failure();
    expr += "(";
    expr += callOp.getCallee();
    expr += " ";
    if (failed(serializeArguments(callOp.getArgOperands(), expr)))
      return failure();
    expr += ")";
    return success();
  }

  // Run custom serializers
  return ::serializeExpression<
      // clang-format off
      ConstantIntOp,
      CmpIOp,
      AddIOp,
      MulIOp
      // clang-format on
      >(parentOp, expr, *this);
}

LogicalResult SMTContext::serializeArguments(ValueRange valueRange,
                                             std::string &expr) {
  bool first = true;
  for (auto val : valueRange) {
    if (!first)
      expr += " ";
    first = false;
    if (failed(serializeExpression(val, expr)))
      return failure();
  }
  return success();
}

LogicalResult SMTContext::serializeStatement(Operation *op, std::string &expr) {
  // If it is not a valid SMT2 statement, just ignore it.
  if (!op->hasTrait<OpTrait::SMTStatement>())
    return success();

  // If it has a serializer interface defined, use it directly.
  if (auto serializer = dyn_cast<SMTSerializableOpInterface>(op)) {
    return serializer.serializeExpression(expr, *this);
  }

  // add custom patterns below:
  // return ::serializeExpression<
  //     // clang-format off
  //     // clang-format on
  //     >(op, expr, *this);
  return op->emitError("[mlir-to-smt] cannot convert operation to SMT "
                       "expression - no registered serialization");
}