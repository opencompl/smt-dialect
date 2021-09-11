#ifndef SMT_OPS
#define SMT_OPS

include "SMTDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Documentation: http://smtlib.cs.uiowa.edu/papers/smt-lib-reference-v2.6-r2021-05-12.pdf

def SMT_AssertOp : SMT_Op<"assert", []> {
  let summary = "(assert <<expr>>)";
  let description = [{
    %b = constant 0 : i1
    smt.assert %b
  }];

  let arguments = (ins I1:$cond);

  let assemblyFormat = [{ $cond attr-dict }];
}

def SMT_CheckSatOp : SMT_Op<"check_sat", []> {
  let summary = "(check-sat)";
  let description = [{
    smt.check_sat
  }];

  let assemblyFormat = [{ attr-dict }];
}

def SMT_GetModelOp : SMT_Op<"get_model", []> {
  let summary = "(get-model)";
  let description = [{
    smt.get_model
  }];

  let assemblyFormat = [{ attr-dict }];
}

#endif // SMT_OPS
