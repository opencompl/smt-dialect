#ifndef SMT_SMTOPS_H
#define SMT_SMTOPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "SMTInterfaces.h"

#define GET_OP_CLASSES
#include "SMT/SMTOps.h.inc"

#include "SMT/SMTOpsDialect.h.inc"

#endif // SMT_SMTOPS_H
