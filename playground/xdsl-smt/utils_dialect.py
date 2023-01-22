from __future__ import annotations
from abc import abstractclassmethod

from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar
from xdsl.ir import (Attribute, MLContext, Operation, ParametrizedAttribute,
                     SSAValue)
from xdsl.irdl import (OperandDef, ParameterDef, ResultDef,
                       irdl_attr_definition, irdl_op_definition, builder)

from dialect import SMTLibOp, SMTLibSort, SimpleSMTLibOp

_F = TypeVar("_F", bound=Attribute)
_S = TypeVar("_S", bound=Attribute)


@irdl_attr_definition
class PairType(Generic[_F, _S], ParametrizedAttribute, SMTLibSort):
    name = "smt.utils.pair"

    first: ParameterDef[_F]
    second: ParameterDef[_S]

    def as_smtlib_str(self) -> str:
        assert isinstance(self.first, SMTLibSort)
        assert isinstance(self.second, SMTLibSort)
        return f"(Pair {self.first.as_smtlib_str()} {self.second.as_smtlib_str()})"

    @builder
    @staticmethod
    def from_params(first: _F, second: _S) -> PairType[_F, _S]:
        return PairType([first, second])


@irdl_op_definition
class PairOp(Operation, SimpleSMTLibOp):
    name = "smt.utils.pair"

    res = ResultDef(PairType)
    first = OperandDef(Attribute)
    second = OperandDef(Attribute)

    @staticmethod
    def from_values(first: SSAValue, second: SSAValue) -> PairOp:
        result_type = PairType.from_params(first.typ, second.typ)
        return PairOp.create(result_types=[result_type],
                             operands=[first, second])

    def op_name(self) -> str:
        return "pair"


@irdl_op_definition
class FirstOp(Operation, SimpleSMTLibOp):
    name = "smt.utils.first"

    res = ResultDef(Attribute)
    pair = OperandDef(PairType)

    @staticmethod
    def from_value(pair: SSAValue) -> FirstOp:
        return FirstOp.create(result_types=[pair.typ.first], operands=[pair])

    def op_name(self) -> str:
        return "first"


@irdl_op_definition
class SecondOp(Operation, SimpleSMTLibOp):
    name = "smt.utils.second"

    res = ResultDef(Attribute)
    pair = OperandDef(PairType)

    def op_name(self) -> str:
        return "second"

    @staticmethod
    def from_value(pair: SSAValue) -> SecondOp:
        return SecondOp.create(result_types=[pair.typ.second], operands=[pair])


@dataclass
class SMTUtilsDialect:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(PairType)
        self.ctx.register_op(PairOp)
        self.ctx.register_op(FirstOp)
        self.ctx.register_op(SecondOp)
