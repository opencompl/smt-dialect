from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from io import IOBase
from typing import TYPE_CHECKING

from xdsl.irdl import (ParameterDef, RegionDef, ResultDef, AttributeDef,
                       irdl_attr_definition, irdl_op_definition, OperandDef)
from xdsl.ir import (MLContext, Operation, Data, ParametrizedAttribute,
                     Attribute)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.builtin import (ArrayAttr, StringAttr)

if TYPE_CHECKING:
    from smt_conversion import SMTConversionCtx


class SMTLibSort:

    @abstractmethod
    def as_smtlib_str(self) -> str:
        ...


@irdl_attr_definition
class BoolType(ParametrizedAttribute, SMTLibSort):
    name = "smt.bool"

    def as_smtlib_str(self) -> str:
        return "Bool"


@irdl_attr_definition
class SortType(ParametrizedAttribute):
    name = "smt.sort"
    identifier: ParameterDef[StringAttr]
    params: ParameterDef[ArrayAttr[Attribute]]  # TODO: ArrayAttr[SortType]


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


# Script operations


@irdl_op_definition
class DeclareConstOp(Operation):
    name = "smt.declare_const"
    res = ResultDef(Attribute)


@irdl_op_definition
class AssertOp(Operation):
    name = "smt.assert"
    op = OperandDef(BoolType)


@irdl_op_definition
class CheckSatOp(Operation):
    name = "smt.check_sat"


# Core operations


@irdl_attr_definition
class BoolAttr(Data[bool]):
    name = "smt.bool_attr"

    @staticmethod
    def parse_parameter(parser: Parser) -> bool:
        if parser.parse_optional_string("true"):
            return True
        if parser.parse_optional_string("false"):
            return False
        raise ValueError("Expected 'true' or 'false'")

    @staticmethod
    def print_parameter(data: bool, printer: Printer) -> None:
        printer.print(str(data))


@irdl_op_definition
class ConstantBoolOp(Operation):
    name = "smt.constant_bool"
    res = ResultDef(BoolType)
    value = AttributeDef(BoolAttr)


@irdl_op_definition
class NotOp(Operation):
    name = "smt.not"
    res = ResultDef(BoolType)
    arg = OperandDef(BoolType)


@irdl_op_definition
class ImpliesOp(Operation):
    name = "smt.implies"
    res = ResultDef(BoolType)
    arg = OperandDef(BoolType)


@irdl_op_definition
class AndOp(Operation):
    name = "smt.and"
    res = ResultDef(BoolType)
    lhs = OperandDef(BoolType)
    rhs = OperandDef(BoolType)


@irdl_op_definition
class OrOp(Operation):
    name = "smt.or"
    res = ResultDef(BoolType)
    lhs = OperandDef(BoolType)
    rhs = OperandDef(BoolType)


@irdl_op_definition
class XorOp(Operation):
    name = "smt.xor"
    res = ResultDef(BoolType)
    lhs = OperandDef(BoolType)
    rhs = OperandDef(BoolType)


@irdl_op_definition
class EqOp(Operation):
    name = "smt.eq"
    res = ResultDef(BoolType)
    lhs = OperandDef(Attribute)
    rhs = OperandDef(Attribute)


@irdl_op_definition
class DiscinctOp(Operation):
    name = "smt.distinct"
    res = ResultDef(BoolType)
    lhs = OperandDef(Attribute)
    rhs = OperandDef(Attribute)


@irdl_op_definition
class IteOp(Operation):
    name = "smt.ite"
    res = ResultDef(Attribute)
    cond = OperandDef(BoolType)
    true_val = OperandDef(Attribute)
    false_val = OperandDef(Attribute)


@dataclass
class SMTDialect:
    ctx: MLContext

    def __post_init__(self):
        # Base SMTLib language
        self.ctx.register_attr(BoolType)
        self.ctx.register_attr(SortType)
        self.ctx.register_op(YieldOp)
        self.ctx.register_op(ForallOp)
        self.ctx.register_op(ExistsOp)

        # SMTLib Scripting
        self.ctx.register_op(DeclareConstOp)
        self.ctx.register_op(AssertOp)
        self.ctx.register_op(CheckSatOp)

        # Core theory
        self.ctx.register_attr(BoolAttr)
        self.ctx.register_op(ConstantBoolOp)
        self.ctx.register_op(NotOp)
        self.ctx.register_op(ImpliesOp)
        self.ctx.register_op(AndOp)
        self.ctx.register_op(OrOp)
        self.ctx.register_op(XorOp)
        self.ctx.register_op(EqOp)
        self.ctx.register_op(DiscinctOp)
        self.ctx.register_op(IteOp)
