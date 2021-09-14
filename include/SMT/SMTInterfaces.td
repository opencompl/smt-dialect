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
      /*desc=*/"Serializes into an SMT2 expression, and appends to the `expr` string",
      /*retTy=*/"LogicalResult",
      /*methodName=*/"serializeExpression",
      /*args=*/(ins "std::string&":$expr, "smt::SMTContext&":$ctx),
      /*methodBody*/[{}]
    >
  ];
}

#endif // SMTOPINTERFACES
