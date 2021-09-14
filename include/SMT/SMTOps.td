#ifndef SMT_OPS
#define SMT_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "SMTInterfaces.td"

//===----------------------------------------------------------------------===//
// SMT dialect definition.
//===----------------------------------------------------------------------===//

def SMT_Dialect : Dialect {
  let name = "smt";
  let summary = "MLIR SMT dialect";
  let description = [{
    [Documentation](http://smtlib.cs.uiowa.edu/papers/smt-lib-reference-v2.6-r2021-05-12.pdf)
  }];
  let cppNamespace = "::mlir::smt";
}

//===----------------------------------------------------------------------===//
// Base SMT operation definition.
//===----------------------------------------------------------------------===//

// SMT operation, which corresponds to an smt2 statement.
class SMT_Op<string mnemonic, list<OpTrait> traits = []> :
  Op<SMT_Dialect, mnemonic,
    !listconcat(traits, [DeclareOpInterfaceMethods<SMTTranslateOpInterface>])>;

// generic support operation (may not be lowerable to smt2)
// could be used to track information etc.
class SMT_SupportOp<string mnemonic, list<OpTrait> traits = []> :
  Op<SMT_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// SMT operations
//===----------------------------------------------------------------------===//

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
