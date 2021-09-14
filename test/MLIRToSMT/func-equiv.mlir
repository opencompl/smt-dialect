// RUN: smt-opt %s
// RUN: smt-translate --mlir-to-smt %s | FileCheck %s
// RUN: smt-translate --mlir-to-smt %s | z3 -in

func @add(%x : i32) -> i32 {
  %s = addi %x, %x : i32
  return %s : i32
}

func @mul(%x : i32) -> i32 {
  %2 = constant 2 : i32
  %s = muli %x, %2 : i32
  return %s : i32
}

func @main() attributes {smt_main} {
  %e = smt.forall (%x : i32) {
    %a = call @add(%x) : (i32) -> i32
    %b = call @mul(%x) : (i32) -> i32
    %e = cmpi "eq", %a, %b : i32
    smt.yield %e : i1
  }
  smt.assert %e
  smt.check_sat
  return
}

// CHECK: (define-fun add (Int) (arg0 Int) (+ arg0 arg0))
// CHECK: (define-fun mul (Int) (arg0 Int) (* arg02))
// CHECK: (assert (forall ((arg0 Int)) (= (add arg0) (mul arg0))))
// CHECK: (check-sat)
