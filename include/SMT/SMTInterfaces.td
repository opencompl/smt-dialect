#ifndef SMTOPINTERFACES
#define SMTOPINTERFACES

include "mlir/IR/OpBase.td"

def SMTSerializableOpInterface : OpInterface<"SMTSerializableOpInterface"> {
  let description = [{
    Can be serialized into an SMT2 expression, using an SMTContext to add
    neccessary function definitions.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Serializes into an SMT2 expression, and appends to the stream `os`",
      /*retTy=*/"LogicalResult",
      /*methodName=*/"serializeExpression",
      /*args=*/(ins "llvm::raw_ostream&":$os, "smt::SMTContext&":$ctx),
      /*methodBody=*/[{}]
    >,

    InterfaceMethod<
      /*desc=*/"Generates the neccessary definition(s) to describe the operation semantics",
      /*retTy=*/"LogicalResult",
      /*methodName=*/"generateDefinitions",
      /*args=*/(ins "smt::SMTContext&":$ctx),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return success(); }]
    >
  ];
}

#endif // SMTOPINTERFACES
