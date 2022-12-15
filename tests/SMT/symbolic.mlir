// RUN: smt-opt %s | smt-opt


// Generate symbolic expressions and extract values.
func @main(%x: i32) attributes {smt_main} {
    %c2 = constant 2 : i32
    %z = "smt.symbol"() : () -> (i32)
    %z_times_two = muli %z, %c2  : i32  // z*2
    %e = cmpi "eq", %x, %z_times_two : i32 // z*2==x
    "smt.assert" %e : (i1) -> ()
    smt.check_sat
    %mdl = smt.get_model // returns a model.
    %zv = smt.get_model_value %mdl, %z // extract z such that z*2=x
    return %zv
}

