#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "SMT/SMTOps.h"
#include "SMT/SMTSerializers.h"

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
// LogicalResult serializeExpression(MyOperation op, raw_ostream &os,
//                                   SMTContext &ctx);
// * Write the serialized expression to `os`.

LogicalResult serializeExpression(arith::ConstantIntOp op,
                                  llvm::raw_ostream &os, SMTContext &ctx) {
  IntegerType intType = op.getType().cast<IntegerType>();
  unsigned width = intType.getWidth();
  int64_t value = op.value();
  if (width == 1) { // Boolean value
    os << (value ? "true" : "false");
    return success();
  }
  if (intType.isUnsigned()) { // unsigned, treat as bitvector.
    os << "(_ bv" << (uint64_t)value << " " << width << ")";
  }
  if (value < 0) {
    os << "(- " << (-value) << ")";
  } else {
    os << value;
  }
  return success();
}

LogicalResult serializeExpression(arith::CmpIOp op, llvm::raw_ostream &os,
                                  SMTContext &ctx) {
  switch (op.getPredicate()) {
  case arith::CmpIPredicate::eq:
    os << "(= ";
    break;
  case arith::CmpIPredicate::ne:
    os << "(not (= ";
    break;
  case arith::CmpIPredicate::sge:
    os << "(>= ";
    break;
  case arith::CmpIPredicate::sgt:
    os << "(> ";
    break;
  case arith::CmpIPredicate::sle:
    os << "(<= ";
    break;
  case arith::CmpIPredicate::slt:
    os << "(< ";
    break;
  default:
    op->emitError("[mlir-to-smt] Unsigned comparisons not supported (because I "
                  "haven't yet figured out how to compare bitvectors).");
  }
  if (failed(ctx.serializeExpression(op.getLhs(), os)))
    return failure();
  os << " ";
  if (failed(ctx.serializeExpression(op.getRhs(), os)))
    return failure();
  os << ")";
  if (op.getPredicate() == arith::CmpIPredicate::ne)
    os << ")";
  return success();
}

template <typename Op>
LogicalResult serializeBinOp(const char *opSym, Op op, llvm::raw_ostream &os,
                             SMTContext &ctx) {
  os << "(" << opSym << " ";
  if (failed(ctx.serializeExpression(op.getLhs(), os)))
    return failure();
  os << " ";
  if (failed(ctx.serializeExpression(op.getRhs(), os)))
    return failure();
  os << ")";
  return success();
}

LogicalResult serializeExpression(arith::AddIOp op, llvm::raw_ostream &os,
                                  SMTContext &ctx) {
  return serializeBinOp<arith::AddIOp>("+", op, os, ctx);
}

LogicalResult serializeExpression(arith::MulIOp op, llvm::raw_ostream &os,
                                  SMTContext &ctx) {
  return serializeBinOp<arith::MulIOp>("*", op, os, ctx);
}

//==== Expression Serializer Execution =======================================//

template <typename Op>
LogicalResult serializeExpression(Operation *op, llvm::raw_ostream &os,
                                  SMTContext &ctx) {
  if (auto opp = dyn_cast<Op>(op)) {
    return serializeExpression(opp, os, ctx);
  }
  return op->emitError(
      "[mlir-to-smt] cannot convert operation to SMT expression - no "
      "registered serialization");
}

template <typename Op, typename T, typename... Ts>
LogicalResult serializeExpression(Operation *op, llvm::raw_ostream &os,
                                  SMTContext &ctx) {
  if (auto opp = dyn_cast<Op>(op)) {
    return serializeExpression(opp, os, ctx);
  }
  return serializeExpression<T, Ts...>(op, os, ctx);
}
} // namespace

//==== SMTContext ===========================================================//
LogicalResult SMTContext::addFunc(StringRef funcName, FunctionType funcType,
                                  std::string body) {
  // TODO: verify that the type signatures match
  if (funcDefs.find(funcName.str()) != funcDefs.end()) {
    return success();
  }

  llvm::raw_string_ostream def(funcDefs[funcName.str()]);
  const bool isDecl = body.empty();

  def << "(" << (isDecl ? "declare-fun" : "define-fun") << " " << funcName;

  // argument list
  def << " (";
  for (auto param : llvm::enumerate(funcType.getInputs())) {
    def << "(arg" << std::to_string(param.index()) << " ";
    if (failed(printGenericType(param.value(), def)))
      return failure();
    def << ")";
  }
  def << ") ";

  // return type
  if (failed(printGenericType(funcType.getResult(0), def)))
    return failure();

  // body (if it exists)
  if (!isDecl)
    def << " " << body;

  def << ")";

  // print the function definition
  primaryOS << def.str() << '\n';

  return success();
}

LogicalResult SMTContext::addMLIRFunction(StringRef funcName) {
  if (auto func = dyn_cast<func::FuncOp>(module.lookupSymbol(funcName))) {
    // found the function declaration in the module, so add that.
    if (func.getFunctionType().getNumResults() != 1) {
      return func.emitError("[mlir-to-smt] only functions with a single return "
                            "value are supported.");
    }

    std::string body;
    if (!func.isDeclaration()) {
      llvm::raw_string_ostream os(body);
      auto &blocks = func.getRegion().getBlocks();

      // TODO: support multi-block functions where the CFG is a DAG.
      if (blocks.size() != 1) {
        return func->emitError(
            "[mlir-to-smt] Only functions with a single block supported.");
      }

      // TODO: serialize top-down, using `let` and `if` to handle branching
      if (failed(serializeExpression(
              cast<func::ReturnOp>(blocks.front().getTerminator())
                  .getOperand(0),
              os)))
        return failure();
    }

    return addFunc(funcName, func.getFunctionType(), body);
  }

  return failure();
}

LogicalResult SMTContext::printGenericType(Type type, llvm::raw_ostream &os) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    if (intType.getWidth() == 1) { // Special case: bool type
      os << "Bool";
    } else if (intType.isUnsigned()) { // treat unsigned as bitvectors
      os << "(_ BitVec " << intType.getWidth() << ")";
    } else { // treat signed as generic integers
      os << "Int";
    }
    return success();
  }
  return emitError(UnknownLoc(), "[mlir-to-smt] Unknown type found: ") << type;
}

LogicalResult SMTContext::serializeExpression(Value value,
                                              llvm::raw_ostream &os) {
  // Value is block argument, generate the name and return.
  // NOTE: block arguments do not have a defining op.
  if (auto blockArg = value.dyn_cast<BlockArgument>()) {
    os << "arg" << blockArg.getArgNumber();
    return success();
  }

  Operation *parentOp = value.getDefiningOp();

  // If it has a serializer interface defined, use it directly.
  if (auto serializer = dyn_cast<SMTSerializableOpInterface>(parentOp)) {
    return serializer.serializeExpression(os, *this);
  }

  // handle function calls separately.
  if (auto callOp = dyn_cast<func::CallOp>(parentOp)) {
    if (failed(this->addMLIRFunction(callOp.getCallee())))
      return failure();
    os << "(" << callOp.getCallee() << " ";
    if (failed(serializeArguments(callOp.getArgOperands(), os)))
      return failure();
    os << ")";
    return success();
  }

  // Run custom serializers
  return ::serializeExpression<
      // clang-format off
      arith::ConstantIntOp,
      arith::CmpIOp,
      arith::AddIOp,
      arith::MulIOp
      // clang-format on
      >(parentOp, os, *this);
}

LogicalResult SMTContext::serializeArguments(ValueRange valueRange,
                                             llvm::raw_ostream &os) {
  bool first = true;
  for (auto val : valueRange) {
    if (!first)
      os << " ";
    first = false;
    if (failed(serializeExpression(val, os)))
      return failure();
  }
  return success();
}

LogicalResult SMTContext::serializeStatement(Operation *op,
                                             llvm::raw_ostream &os) {
  // If it is not a valid SMT2 statement, just ignore it.
  if (!op->hasTrait<OpTrait::SMTStatement>())
    return success();

  // If it has a serializer interface defined, use it directly.
  if (auto serializer = dyn_cast<SMTSerializableOpInterface>(op)) {
    return serializer.serializeExpression(os, *this);
  }

  // add custom patterns below:
  // return ::serializeExpression<
  //     // clang-format off
  //     // clang-format on
  //     >(op, expr, *this);
  return op->emitError("[mlir-to-smt] cannot convert operation to SMT "
                       "expression - no registered serialization");
}