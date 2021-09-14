// RUN: smt-opt %s
// RUN: smt-translate --mlir-to-smt %s | FileCheck %s
// RUN: smt-translate --mlir-to-smt %s | z3 -in

func @main() attributes {smt_main} {
  %e = smt.forall (%x : i32) {
    %2 = constant 2 : i32
    %a = addi %x, %x : i32
    %b = muli %x, %2 : i32
    %e = cmpi "eq", %a, %b : i32
    smt.yield %e : i1
  }
  smt.assert %e
  smt.check_sat
  return
}

// CHECK: (assert (forall ((x Int)) (= (+ x x) (* x 2))))
// CHECK: (check-sat)
