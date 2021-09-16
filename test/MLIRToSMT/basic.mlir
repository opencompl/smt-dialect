// RUN: smt-opt %s
// RUN: smt-translate --mlir-to-smt %s | FileCheck %s
// no-run: smt-translate --mlir-to-smt %s | z3 -in

func @main() attributes {smt_main} {
  %x = constant 42 : i32
  %y = constant 0 : i32
  %e = cmpi "sgt", %x, %y : i32
  smt.assert %e
  smt.check_sat
  smt.get_model
  return
}

// CHECK: (assert (> 42 0))
// CHECK: (check-sat)
// CHECK: (get-model)
