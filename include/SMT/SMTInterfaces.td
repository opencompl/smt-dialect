#ifndef SMTOPINTERFACES
#define SMTOPINTERFACES

include "mlir/IR/OpBase.td"

def SMTTranslateOpInterface : OpInterface<"SMTTranslateOpInterface"> {
  let description = [{
    Translate the MLIR Op to an SMT2 Expression.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Prints the SMT Expression to output stream",
      /*retTy=*/"LogicalResult",
      /*methodName=*/"translateToSMT",
      /*args=*/(ins "raw_ostream&":$output),
      /*methodBody*/[{}],
      /*defaultImplementation=*/ [{
        return this->emitWarning("Skipping operation, translation to SMT Expression not provided.");
      }]
    >
  ];
}

#endif // SMTOPINTERFACES
