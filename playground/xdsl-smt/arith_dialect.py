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
    name = "arith.addf"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Addi(Operation):
    name = "arith.addi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class AdduiCarry(Operation):
    name = "arith.addui_carry"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    sum = ResultDef(AnyAttr())
    carry = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Andi(Operation):
    name = "arith.andi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Bitcast(Operation):
    name = "arith.bitcast"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Ceildivsi(Operation):
    name = "arith.ceildivsi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Ceildivui(Operation):
    name = "arith.ceildivui"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Cmpf(Operation):
    name = "arith.cmpf"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Cmpi(Operation):
    name = "arith.cmpi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Constant(Operation):
    name = "arith.constant"
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Divf(Operation):
    name = "arith.divf"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Divsi(Operation):
    name = "arith.divsi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Divui(Operation):
    name = "arith.divui"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Extf(Operation):
    name = "arith.extf"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Extsi(Operation):
    name = "arith.extsi"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Extui(Operation):
    name = "arith.extui"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Fptosi(Operation):
    name = "arith.fptosi"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Fptoui(Operation):
    name = "arith.fptoui"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Floordivsi(Operation):
    name = "arith.floordivsi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class IndexCast(Operation):
    name = "arith.index_cast"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class IndexCastui(Operation):
    name = "arith.index_castui"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Maxf(Operation):
    name = "arith.maxf"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Maxsi(Operation):
    name = "arith.maxsi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Maxui(Operation):
    name = "arith.maxui"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Minf(Operation):
    name = "arith.minf"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Minsi(Operation):
    name = "arith.minsi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Minui(Operation):
    name = "arith.minui"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Mulf(Operation):
    name = "arith.mulf"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Muli(Operation):
    name = "arith.muli"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Negf(Operation):
    name = "arith.negf"
    operand = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Ori(Operation):
    name = "arith.ori"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Remf(Operation):
    name = "arith.remf"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Remsi(Operation):
    name = "arith.remsi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Remui(Operation):
    name = "arith.remui"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Sitofp(Operation):
    name = "arith.sitofp"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Shli(Operation):
    name = "arith.shli"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Shrsi(Operation):
    name = "arith.shrsi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Shrui(Operation):
    name = "arith.shrui"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Subf(Operation):
    name = "arith.subf"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Subi(Operation):
    name = "arith.subi"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Truncf(Operation):
    name = "arith.truncf"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Trunci(Operation):
    name = "arith.trunci"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Uitofp(Operation):
    name = "arith.uitofp"
    _in = OperandDef(AnyAttr())
    out = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Xori(Operation):
    name = "arith.xori"
    lhs = OperandDef(AnyAttr())
    rhs = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()


@irdl_op_definition
class Select(Operation):
    name = "arith.select"
    condition = OperandDef(AnyAttr())
    true_value = OperandDef(AnyAttr())
    false_value = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())
    regs = VarRegionDef()
