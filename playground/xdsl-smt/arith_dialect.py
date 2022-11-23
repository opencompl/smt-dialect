from dataclasses import dataclass
from xdsl.ir import Operation, MLContext
from xdsl.irdl import (OperandDef, ResultDef, AnyAttr, VarRegionDef,
                       irdl_op_definition)


@dataclass
class Arith:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(Addf)
        self.ctx.register_op(Addi)
        self.ctx.register_op(AdduiCarry)
        self.ctx.register_op(Andi)
        self.ctx.register_op(Bitcast)
        self.ctx.register_op(Ceildivsi)
        self.ctx.register_op(Ceildivui)
        self.ctx.register_op(Cmpf)
        self.ctx.register_op(Cmpi)
        self.ctx.register_op(Constant)
        self.ctx.register_op(Divf)
        self.ctx.register_op(Divsi)
        self.ctx.register_op(Divui)
        self.ctx.register_op(Extf)
        self.ctx.register_op(Extsi)
        self.ctx.register_op(Extui)
        self.ctx.register_op(Fptosi)
        self.ctx.register_op(Fptoui)
        self.ctx.register_op(Floordivsi)
        self.ctx.register_op(IndexCast)
        self.ctx.register_op(IndexCastui)
        self.ctx.register_op(Maxf)
        self.ctx.register_op(Maxsi)
        self.ctx.register_op(Maxui)
        self.ctx.register_op(Minf)
        self.ctx.register_op(Minsi)
        self.ctx.register_op(Minui)
        self.ctx.register_op(Mulf)
        self.ctx.register_op(Muli)
        self.ctx.register_op(Negf)
        self.ctx.register_op(Ori)
        self.ctx.register_op(Remf)
        self.ctx.register_op(Remsi)
        self.ctx.register_op(Remui)
        self.ctx.register_op(Sitofp)
        self.ctx.register_op(Shli)
        self.ctx.register_op(Shrsi)
        self.ctx.register_op(Shrui)
        self.ctx.register_op(Subf)
        self.ctx.register_op(Subi)
        self.ctx.register_op(Truncf)
        self.ctx.register_op(Trunci)
        self.ctx.register_op(Uitofp)
        self.ctx.register_op(Xori)
        self.ctx.register_op(Select)


@irdl_op_definition
class Addf(Operation):
    name = "addf"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Addi(Operation):
    name = "addi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class AdduiCarry(Operation):
    name = "addui_carry"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    sum = ResultDef(AnyAttr())
    carry = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Andi(Operation):
    name = "andi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Bitcast(Operation):
    name = "bitcast"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Ceildivsi(Operation):
    name = "ceildivsi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Ceildivui(Operation):
    name = "ceildivui"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Cmpf(Operation):
    name = "cmpf"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Cmpi(Operation):
    name = "cmpi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Constant(Operation):
    name = "constant"
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Divf(Operation):
    name = "divf"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Divsi(Operation):
    name = "divsi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Divui(Operation):
    name = "divui"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Extf(Operation):
    name = "extf"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Extsi(Operation):
    name = "extsi"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Extui(Operation):
    name = "extui"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Fptosi(Operation):
    name = "fptosi"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Fptoui(Operation):
    name = "fptoui"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Floordivsi(Operation):
    name = "floordivsi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class IndexCast(Operation):
    name = "index_cast"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class IndexCastui(Operation):
    name = "index_castui"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Maxf(Operation):
    name = "maxf"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Maxsi(Operation):
    name = "maxsi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Maxui(Operation):
    name = "maxui"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Minf(Operation):
    name = "minf"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Minsi(Operation):
    name = "minsi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Minui(Operation):
    name = "minui"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Mulf(Operation):
    name = "mulf"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Muli(Operation):
    name = "muli"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Negf(Operation):
    name = "negf"
    operand = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Ori(Operation):
    name = "ori"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Remf(Operation):
    name = "remf"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Remsi(Operation):
    name = "remsi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Remui(Operation):
    name = "remui"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Sitofp(Operation):
    name = "sitofp"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Shli(Operation):
    name = "shli"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Shrsi(Operation):
    name = "shrsi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Shrui(Operation):
    name = "shrui"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Subf(Operation):
    name = "subf"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Subi(Operation):
    name = "subi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Truncf(Operation):
    name = "truncf"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Trunci(Operation):
    name = "trunci"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Uitofp(Operation):
    name = "uitofp"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Xori(Operation):
    name = "xori"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Select(Operation):
    name = "select"
    condition = OperandDef(AnyAttr())
    true_value = OperandDef(AnyAttr())
    false_value = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()
