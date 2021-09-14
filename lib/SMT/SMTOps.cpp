#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/FormatVariadic.h"

#include "SMT/SMTOps.h"

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

static LogicalResult verify(ForallOp op) { return success(); }

} // namespace

#define GET_OP_CLASSES
#include "SMT/SMTOps.cpp.inc"
