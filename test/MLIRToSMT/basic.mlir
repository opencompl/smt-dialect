// RUN: smt-opt %s
// RUN: smt-translate --mlir-to-smt %s | FileCheck %s
// RUN: smt-translate --mlir-to-smt %s | z3

func @main() attributes {smt_main} {
  %x = constant 42 : i32
  %y = constant 0 : i32
  %e = cmpi "neq", %x, %y : i32
  smt.assert %e
  smt.check_sat
  smt.get_model
}

// CHECK: (assert (not (= 42 0)))
// CHECK: (check-sat)
// CHECK: (get-model)
