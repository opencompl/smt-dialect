#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "SMT/SMTOps.h"
#include "Target/MLIRToSMT.h"

using namespace mlir;
using namespace smt;

namespace {

//==== Operation Serializers ================================================//

// Registering a serializer:
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

//==== Serializer Executor ================================================//
template <typename Op>
LogicalResult serializeExpression(Operation *op, std::string &expr,
                                  SMTContext &ctx) {
  if (auto opp = dyn_cast<Op>(op)) {
    return serializeExpression(opp, expr, ctx);
  }
  op->emitError("[mlir-to-smt] cannot convert operation to SMT expression - no "
                "registered serialization");
  return failure();
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

LogicalResult SMTContext::serializeExpression(Value value, std::string &expr) {
  Operation *op = value.getDefiningOp();
  return ::serializeExpression<
      // clang-format off
      ConstantIntOp,
      CmpIOp
      // clang-format on
      >(op, expr, *this);
}