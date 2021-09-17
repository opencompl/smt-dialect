// RUN: smt-opt %s | smt-opt


// Run a predicated optimization. If exists a z such that z * 2 = x, then return 0, else 1.
// This a really complicated way to implement (x % 2 == 1) by encoding as 
// ((exists z . z * 2 == x) ? 0 : 1)
func @check_odd() -> i1 attributes {smt_main} {
    %x = constant 42 : i32
    %c2 = constant 2 : i32
    %z = "smt.symbol"() : () -> i32
    %z_times_two = muli %z, %c2  : i32  // z*2
    %e = cmpi "eq", %x, %z_times_two : i32 // z*2==x
    "smt.assert"(%e) : (i1) -> ()
    smt.check_sat
    %is_sat = "smt.is_sat"() : () -> (i1)
    %f = constant 0 : i1
    %is_not_st = cmpi "eq", %is_sat, %f : i1
    return %is_not_st : i1
}

