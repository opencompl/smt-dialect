#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/FormatVariadic.h"

#include "SMT/SMTOps.h"

#define GET_OP_CLASSES
#include "SMT/SMTOps.cpp.inc"

using namespace mlir;
using namespace smt;

LogicalResult AssertOp::translateToSMT(raw_ostream &output) {
  return failure();
}

LogicalResult CheckSatOp::translateToSMT(raw_ostream &output) {
  output << "(check-sat)\n";
  return success();
}

LogicalResult GetModelOp::translateToSMT(raw_ostream &output) {
  output << "(get-model)\n";
  return success();
}
