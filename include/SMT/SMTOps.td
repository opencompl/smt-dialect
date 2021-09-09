#ifndef SMT_OPS
#define SMT_OPS

include "SMTDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SMT_FooOp : SMT_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
      dummy operation
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // SMT_OPS
