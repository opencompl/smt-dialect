from dataclasses import dataclass
from typing import Generic, TypeVar
from xdsl.ir import Attribute, MLContext, Operation, ParametrizedAttribute
from xdsl.irdl import AttributeDef, OperandDef, ParameterDef, ResultDef, irdl_attr_definition, irdl_op_definition

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


@irdl_op_definition
class ConstantPairOp(Operation, SMTLibOp):
    name = "smt.utils.constant_pair"

    res = ResultDef(PairType)
    first = OperandDef(Attribute)
    second = OperandDef(Attribute)

    def as_smtlib_str(self) -> str:
        return "pair"


@irdl_op_definition
class FirstOp(Operation, SimpleSMTLibOp):
    name = "smt.utils.first"

    res = ResultDef(Attribute)
    pair = OperandDef(PairType)

    def op_name(self) -> str:
        return "first"


@irdl_op_definition
class SecondOp(Operation, SimpleSMTLibOp):
    name = "smt.utils.second"

    res = ResultDef(Attribute)
    pair = OperandDef(PairType)

    def op_name(self) -> str:
        return "second"


@dataclass
class SMTUtilsDialect:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(PairType)
        self.ctx.register_op(ConstantPairOp)
        self.ctx.register_op(FirstOp)
        self.ctx.register_op(SecondOp)
