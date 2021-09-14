#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/FormatVariadic.h"

#include "SMT/SMTOps.h"

#define GET_OP_CLASSES
#include "SMT/SMTOps.cpp.inc"
