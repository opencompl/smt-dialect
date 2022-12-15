// RUN: smt-opt %s | smt-opt

// (declare-fun y () Int)
// 
// (assert (= 42 y))
// (check-sat)
// (get-model)

func private @y() -> i32

func @main() attributes {smt_main} {
	%x = constant 42 : i32
	%y = call @y() : () -> i32
	%e = cmpi "eq", %x, %y : i32
	smt.assert %e
	smt.check_sat
	smt.get_model
	return
}