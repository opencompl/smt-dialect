from __future__ import annotations
from dataclasses import dataclass

from xdsl.irdl import ParameterDef, RegionDef, ResultDef, irdl_attr_definition, irdl_op_definition
from xdsl.ir import MLContext, Operation, ParametrizedAttribute
from xdsl.dialects.builtin import ArrayAttr, StringAttr


@irdl_attr_definition
class BoolType(ParametrizedAttribute):
    name = "smt.bool"


@irdl_attr_definition
class SortType(ParametrizedAttribute):
    name = "smt.sort"
    identifier: ParameterDef[StringAttr]
    params: ParameterDef[ArrayAttr[SortType]]


@irdl_op_definition
class YieldOp(Operation):
    name = "smt.yield"
    ret = ResultDef(BoolType)


@irdl_op_definition
class ForallOp(Operation):
    name = "smt.forall"
    res = ResultDef(BoolType)
    reg = RegionDef()


@irdl_op_definition
class ExistsOp(Operation):
    name = "smt.exists"
    res = ResultDef(BoolType)
    reg = RegionDef()


@dataclass
class SMTDialect:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(BoolType)
        self.ctx.register_attr(SortType)
        self.ctx.register_op(YieldOp)
        self.ctx.register_op(ForallOp)
        self.ctx.register_op(ExistsOp)
