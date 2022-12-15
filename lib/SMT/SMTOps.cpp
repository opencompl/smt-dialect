#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "SMT/SMTOps.h"
#include "SMT/SMTSerializers.h"
#include "Target/MLIRToSMT.h"

using namespace mlir;
using namespace smt;

ParseResult ForallOp::parse(OpAsmParser &parser, OperationState &result) {
  // parse the region argument list (i.e. the forall variables)
  SmallVector<OpAsmParser::Argument> args;
  if (failed(parser.parseArgumentList(args,
                                      /*delimiter=*/AsmParser::Delimiter::Paren,
                                      /*allowAttrs=*/true)))
    return failure();

  Region *body = result.addRegion();
  if (failed(parser.parseRegion(*body, args)))
    return failure();

  result.addTypes({parser.getBuilder().getI1Type()});

  return success();
}

void ForallOp::print(OpAsmPrinter &printer) {
  printer << "(";
  llvm::interleaveComma(
      getBody().getArguments(), printer,
      [&](BlockArgument v) { printer.printRegionArgument(v); });
  printer << ") ";
  // Get the attribute dictionary
  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer.printRegion(getBody(),
                      /*printEntryBlockArgs=*/false);
}

#define GET_OP_CLASSES
#include "SMT/SMTOps.cpp.inc"

YieldOp ForallOp::getYield() {
  return cast<YieldOp>(getBody().getBlocks().front().getTerminator());
}

//=== Serialize Expression Interface Methods =================================//

LogicalResult ForallOp::serializeExpression(llvm::raw_ostream &os,
                                            smt::SMTContext &ctx) {
  os << "(forall (";

  bool first = true;
  for (auto arg : getBody().getArguments()) {
    if (!first)
      os << " ";
    first = false;
    os << "(arg" << arg.getArgNumber() << " ";
    if (failed(ctx.printGenericType(arg.getType(), os)))
      return failure();
    os << ")";
  }
  os << ") ";
  if (failed(ctx.serializeExpression(getYield().getOperand(0), os)))
    return failure();
  os << ")";
  return success();
}

LogicalResult AssertOp::serializeExpression(llvm::raw_ostream &os,
                                            smt::SMTContext &ctx) {
  os << "(assert ";
  if (failed(ctx.serializeExpression(getCond(), os)))
    return failure();
  os << ")";
  return success();
}

LogicalResult CheckSatOp::serializeExpression(llvm::raw_ostream &os,
                                              smt::SMTContext &ctx) {
  os << "(check-sat)";
  return success();
}
LogicalResult GetModelOp::serializeExpression(llvm::raw_ostream &os,
                                              smt::SMTContext &ctx) {
  os << "(get-model)";
  return success();
}
