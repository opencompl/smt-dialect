#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "SMT/SMTOps.h"
#include "Target/MLIRToSMT.h"

using namespace mlir;
using namespace smt;

namespace {
static LogicalResult parseForallOp(OpAsmParser &parser,
                                   OperationState &result) {
  // parse the region argument list (i.e. the forall variables)
  if (failed(parser.parseLParen()))
    return failure();
  SmallVector<OpAsmParser::OperandType> args;
  SmallVector<Type> argTypes;
  bool first = true;
  while (true) {
    if (!first && failed(parser.parseOptionalComma()))
      break;
    first = false;

    OpAsmParser::OperandType arg;
    Type ty;
    if (failed(parser.parseRegionArgument(arg)) ||
        failed(parser.parseColonType(ty)))
      return failure();
    args.push_back(arg);
    argTypes.push_back(ty);
  }
  if (failed(parser.parseRParen()))
    return failure();

  Region *body = result.addRegion();
  if (failed(parser.parseRegion(*body, args, argTypes)))
    return failure();

  result.addTypes({parser.getBuilder().getI1Type()});

  return success();
}

static void print(OpAsmPrinter &printer, ForallOp op) {
  printer << "(";
  llvm::interleaveComma(op.body().getArguments(), printer,
                        [&](Value v) { printer << v << " : " << v.getType(); });
  printer << ") ";
  printer.printOptionalAttrDict(op->getAttrs());
  printer.printRegion(op.body());
}

} // namespace

#define GET_OP_CLASSES
#include "SMT/SMTOps.cpp.inc"

YieldOp ForallOp::getYield() {
  return cast<YieldOp>(body().getBlocks().front().getTerminator());
}

//=== Serialize Expression Interface Methods =================================//

LogicalResult ForallOp::serializeExpression(std::string &expr,
                                            smt::SMTContext &ctx) {
  llvm::raw_string_ostream os(expr);
  os << "(forall (";
  llvm::interleave(
      body().getArguments(),
      [&](BlockArgument arg) {
        os << "(" << arg << " " << arg.getType() << ")";
      },
      [&]() { os << " "; });
  os << ")";
  return success();
}

LogicalResult AssertOp::serializeExpression(std::string &expr,
                                            smt::SMTContext &ctx) {
  expr += "(assert ";
  if (failed(ctx.serializeExpression(cond(), expr)))
    return failure();
  expr += ")";
  return success();
}

LogicalResult CheckSatOp::serializeExpression(std::string &expr,
                                              smt::SMTContext &ctx) {
  expr += "(check-sat)";
  return success();
}
LogicalResult GetModelOp::serializeExpression(std::string &expr,
                                              smt::SMTContext &ctx) {
  expr += "(get-model)";
  return success();
}
