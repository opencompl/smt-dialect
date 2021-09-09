#include "SMT/SMTDialect.h"
#include "SMT/SMTOps.h"

using namespace mlir;
using namespace mlir::smt;

//===----------------------------------------------------------------------===//
// SMT dialect.
//===----------------------------------------------------------------------===//

void SMTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SMT/SMTOps.cpp.inc"
      >();
}
