from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.arith import MLContext
from xdsl.dialects.builtin import StringAttr, IntAttr
from xdsl.ir import Operation, ParametrizedAttribute
from xdsl.irdl import ParameterDef, irdl_op_definition, irdl_attr_definition, AttributeDef, ResultDef, builder
from dialect import ConstantBoolOp, SMTLibOp, SMTLibSort
from xdsl.parser import Parser
from xdsl.printer import Printer


@irdl_attr_definition
class BitVectorType(ParametrizedAttribute, SMTLibSort):
    name = "smt.bv.bv"
    width: ParameterDef[IntAttr]

    def as_smtlib_str(self) -> str:
        return f"(_ BitVec {self.width.data})"

    @builder
    @staticmethod
    def from_int(value: int) -> BitVectorType:
        return BitVectorType([IntAttr.build(value)])

    @staticmethod
    def parse_parameters(parser: Parser) -> list[Attribute]:
        parser.parse_char("<")
        width = parser.parse_int_literal()
        parser.parse_char(">")
        return [IntAttr.build(width)]

    def print_parameters(self, printer: Printer) -> None:
        printer.print("<", self.width.data, ">")


@irdl_attr_definition
class BitVectorValue(ParametrizedAttribute):
    name = "smt.bv.bv_val"

    value: ParameterDef[IntAttr]
    width: ParameterDef[IntAttr]

    def verify(self) -> None:
        if not (0 <= self.value.data < 2**self.width.data):
            raise ValueError("BitVector value out of range")

    def as_smtlib_str(self) -> str:
        return f"(_ bv{self.value.data} {self.width.data})"

    @staticmethod
    def parse_parameters(parser: Parser) -> list[Attribute]:
        parser.parse_char("<")
        value = parser.parse_int_literal()
        parser.parse_char(",")
        width = parser.parse_int_literal()
        parser.parse_char(">")
        return [IntAttr.build(value), IntAttr.build(width)]

    def print_parameters(self, printer: Printer) -> None:
        printer.print("<", self.value.data, ", ", self.width.data, ">")


@irdl_op_definition
class ConstantOp(Operation, SMTLibOp):
    name = "smt.bv.constant"
    value = AttributeDef(BitVectorValue)
    res = ResultDef(BitVectorType)

    @classmethod
    def parse(cls, result_types: list[Attribute],
              parser: Parser) -> ConstantBoolOp:
        attr = parser.parse_attribute()
        if not isinstance(attr, BitVectorValue):
            raise ValueError("Expected a bitvector value")
        return ConstantOp.create(result_types=[BitVectorType([attr.width])],
                                 attributes={'value': attr})

    def print(self, printer: Printer) -> None:
        printer.print(" ", self.value)

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        print(self.value.as_smtlib_str(), file=stream, end='')


@dataclass
class SMTBitVectorDialect:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(BitVectorType)
        self.ctx.register_attr(BitVectorValue)
        self.ctx.register_op(ConstantOp)