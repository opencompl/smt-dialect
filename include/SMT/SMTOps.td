#ifndef SMT_OPS
#define SMT_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
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
// Types and Traits 
//===----------------------------------------------------------------------===//

def SMTStatement : NativeOpTrait<"SMTStatement">;

//===----------------------------------------------------------------------===//
// Base SMT operation definition.
//===----------------------------------------------------------------------===//


// SMT operation, which corresponds to an smt2 statement.
class SMT_Op<string mnemonic, list<OpTrait> traits = []> :
  Op<SMT_Dialect, mnemonic,
    !listconcat(traits, [DeclareOpInterfaceMethods<SMTSerializableOpInterface>])> {
  let parser = "return ::parse$cppClass(parser, result);";
  let printer = "return ::print(p, *this);";
  let verifier = "return ::verify(*this);";
}

// generic support operation (may not be lowerable to smt2)
// could be used to track information etc.
class SMT_SupportOp<string mnemonic, list<OpTrait> traits = []> :
  Op<SMT_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// SMT operations
//===----------------------------------------------------------------------===//

// Reference Implementation: scf::ForOp
// https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SCF/SCFOps.td#L117
def SMT_ForallOp : SMT_Op<"forall", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "(forall (<<arg>>*) (<<expr>>))";
  let description = [{
    %e = smt.forall (%x : i32) {
      %2 = constant 2 : i32
      %a = addi %x, %x : i32
      %b = muli %x, %2 : i32
      %e = cmpi "eq", %a, %b : i32
      smt.yield %e : i1
    }
    smt.assert %e
  }];

  let regions = (region SizedRegion<1>:$body);
  let results = (outs I1:$cond);
  let verifier = ?;

  let extraClassDeclaration = [{
    YieldOp getYield();
  }];
}

def SMT_YieldOp : SMT_SupportOp<"yield", [Terminator, ReturnLike, ParentOneOf<["ForallOp"]>]> {
  let summary = "Return a value to the parent operation";
  let description = [{
    smt.yield %a, %b : i1, i32
  }];

  let arguments = (ins Variadic<AnyType>:$params);
  let assemblyFormat = [{ $params attr-dict `:` type($params) }];
  let verifier = ?;
}

def SMT_AssertOp : SMT_Op<"assert", [SMTStatement]> {
  let summary = "(assert <<expr>>)";
  let description = [{
    %b = constant 0 : i1
    smt.assert %b
  }];

  let arguments = (ins I1:$cond);

  let assemblyFormat = [{ $cond attr-dict }];
  let verifier = ?;
}

def SMT_CheckSatOp : SMT_Op<"check_sat", [SMTStatement]> {
  let summary = "(check-sat)";
  let description = [{
    smt.check_sat
  }];

  let assemblyFormat = [{ attr-dict }];
  let verifier = ?;
}

def SMT_GetModelOp : SMT_Op<"get_model", [SMTStatement]> {
  let summary = "(get-model)";
  let description = [{
    smt.get_model
  }];

  let assemblyFormat = [{ attr-dict }];
  let verifier = ?;
}

#endif // SMT_OPS
