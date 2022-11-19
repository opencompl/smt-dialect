from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from io import IOBase
from typing import TYPE_CHECKING, TypeVar

from xdsl.irdl import (OptAttributeDef, ParameterDef, RegionDef, ResultDef,
                       AttributeDef, irdl_attr_definition, irdl_op_definition,
                       OperandDef, VarOperandDef)
from xdsl.ir import (MLContext, Operation, Data, ParametrizedAttribute,
                     Attribute, SSAValue)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.builtin import (ArrayAttr, FunctionType, StringAttr)

if TYPE_CHECKING:
    from smt_conversion import SMTConversionCtx


class SMTLibSort:

    @abstractmethod
    def as_smtlib_str(self) -> str:
        ...


class SMTLibOp:

    @abstractmethod
    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        ...


class SMTLibScriptOp(SMTLibOp):
    pass


class SimpleSMTLibOp(SMTLibOp):

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        assert isinstance(self, Operation)
        print(f"({self.op_name()}", file=stream, end='')
        for operand in self.operands:
            print(" ", file=stream, end='')
            ctx.print_expr_to_smtlib(operand, stream)
        print(")", file=stream, end='')

    @abstractmethod
    def op_name(self) -> str:
        ...


@irdl_attr_definition
class BoolType(ParametrizedAttribute, SMTLibSort):
    name = "smt.bool"

    def as_smtlib_str(self) -> str:
        return "Bool"


@irdl_op_definition
class YieldOp(Operation):
    name = "smt.yield"
    ret = OperandDef(BoolType)

    @classmethod
    def parse(cls: type[YieldOp], result_types: list[Attribute],
              parser: Parser) -> YieldOp:
        ret = parser.parse_ssa_value()
        return YieldOp.create(operands=[ret])

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_ssa_value(self.ret)


@irdl_op_definition
class ForallOp(Operation, SMTLibOp):
    name = "smt.forall"
    res = ResultDef(BoolType)
    body = RegionDef()

    @property
    def return_val(self) -> SSAValue:
        yield_op = self.body.ops[-1]
        if not isinstance(yield_op, YieldOp):
            raise ValueError("Region does not end in yield")
        return yield_op.ret

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        print("(forall (", file=stream, end='')
        for idx, param in enumerate(self.body.blocks[0].args):
            param_name = ctx.get_fresh_name(param)
            if idx != 0:
                print(" ", file=stream, end='')
            print(f"({param_name} {param.typ.as_smtlib_str()})",
                  file=stream,
                  end='')
        print(") ", file=stream, end='')
        ctx.print_expr_to_smtlib(self.return_val, stream)
        print(")", file=stream, end='')


@irdl_op_definition
class ExistsOp(Operation, SMTLibOp):
    name = "smt.exists"
    res = ResultDef(BoolType)
    body = RegionDef()

    @property
    def return_val(self) -> SSAValue:
        yield_op = self.body.ops[-1]
        if not isinstance(yield_op, YieldOp):
            raise ValueError("Region does not end in yield")
        return yield_op.ret

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        print("(exists (", file=stream, end='')
        for idx, param in enumerate(self.body.blocks[0].args):
            param_name = ctx.get_fresh_name(param)
            if idx != 0:
                print(" ", file=stream, end='')
            print(f"({param_name} {param.typ.as_smtlib_str()})",
                  file=stream,
                  end='')
        print(") ", file=stream, end='')
        ctx.print_expr_to_smtlib(self.return_val, stream)
        print(")", file=stream, end='')


@irdl_op_definition
class CallOp(Operation, SMTLibOp):
    name = "smt.call"
    res = ResultDef(Attribute)
    func = OperandDef(Attribute)
    args = VarOperandDef(Attribute)

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        print("(", file=stream, end='')
        for idx, operand in enumerate(self.operands):
            if idx != 0:
                print(" ", file=stream, end='')
            ctx.print_expr_to_smtlib(operand, stream)
        print(")", file=stream, end='')


# Script operations


@irdl_op_definition
class DefineFunOp(Operation, SMTLibScriptOp):
    name = "smt.define_fun"
    fun_name = OptAttributeDef(StringAttr)
    ret = ResultDef(FunctionType)
    body = RegionDef()

    @property
    def func_type(self) -> FunctionType:
        return self.ret.typ

    @property
    def return_val(self) -> SSAValue:
        yield_op = self.body.ops[-1]
        if not isinstance(yield_op, YieldOp):
            raise ValueError("Region does not end in yield")
        return yield_op.ret

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        print("(define-fun ", file=stream, end='')

        # Print the function name
        name: str
        if self.fun_name is not None:
            name = ctx.get_fresh_name(self.fun_name.data)
            ctx.value_to_name[self.ret] = name
        else:
            name = ctx.get_fresh_name(self.ret)
        print(f"{name} ", file=stream, end='')

        # Print the function arguments
        print("(", file=stream, end='')
        for idx, arg in enumerate(self.body.blocks[0].args):
            if idx != 0:
                print(" ", file=stream, end='')
            arg_name = ctx.get_fresh_name(arg)
            typ = arg.typ
            assert isinstance(typ, SMTLibSort)
            print(f"({arg_name} {typ.as_smtlib_str()})", file=stream, end='')
        print(") ", file=stream, end='')

        # Print the function return type
        # TODO: support multiple results, or assert this in the verifier
        assert (len(self.func_type.outputs.data) == 1)
        ret_type = self.func_type.outputs.data[0]
        assert isinstance(ret_type, SMTLibSort)
        print(f"{ret_type.as_smtlib_str()}", file=stream)

        # Print the function body
        print("  ", file=stream, end='')
        ctx.print_expr_to_smtlib(self.return_val, stream)
        print(")", file=stream)


@irdl_op_definition
class DeclareConstOp(Operation, SMTLibScriptOp):
    name = "smt.declare_const"
    res = ResultDef(Attribute)

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        name = ctx.get_fresh_name(self.res)
        typ = self.res.typ
        assert isinstance(typ, SMTLibSort)
        print(f"(declare-const {name} {typ.as_smtlib_str()})", file=stream)

    @classmethod
    def parse(cls: type[DeclareConstOp], result_types: list[Attribute],
              parser: Parser) -> DeclareConstOp:
        return DeclareConstOp.create(result_types=result_types)

    def print(self, printer: Printer):
        pass


@irdl_op_definition
class AssertOp(Operation, SMTLibScriptOp):
    name = "smt.assert"
    op = OperandDef(BoolType)

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        print("(assert ", file=stream, end='')
        ctx.print_expr_to_smtlib(self.op, stream)
        print(")", file=stream)

    @classmethod
    def parse(cls: type[AssertOp], result_types: list[Attribute],
              parser: Parser) -> AssertOp:
        operand = parser.parse_ssa_value()
        return AssertOp.create(operands=[operand])

    def print(self, printer: Printer):
        printer.print(" ", self.op)


@irdl_op_definition
class CheckSatOp(Operation, SMTLibScriptOp):
    name = "smt.check_sat"

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        print("(check-sat)", file=stream)

    @classmethod
    def parse(cls: type[CheckSatOp], result_types: list[Attribute],
              parser: Parser) -> CheckSatOp:
        return CheckSatOp.create()

    def print(self, printer: Printer):
        pass


# Core operations

_OpT = TypeVar("_OpT", bound=Operation)


class BinaryBoolOp(Operation):
    res = ResultDef(BoolType)
    lhs = OperandDef(BoolType)
    rhs = OperandDef(BoolType)

    @classmethod
    def parse(cls: type[_OpT], result_types: list[Attribute],
              parser: Parser) -> _OpT:
        lhs = parser.parse_ssa_value()
        parser.parse_string(",")
        rhs = parser.parse_ssa_value()
        return cls.build(result_types=[BoolType([])], operands=[lhs, rhs])

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_ssa_value(self.lhs)
        printer.print(", ")
        printer.print_ssa_value(self.rhs)


class BinaryTOp(Operation):
    res = ResultDef(BoolType)
    lhs = OperandDef(Attribute)
    rhs = OperandDef(Attribute)

    @classmethod
    def parse(cls: type[_OpT], result_types: list[Attribute],
              parser: Parser) -> _OpT:
        lhs = parser.parse_ssa_value()
        parser.parse_string(",")
        rhs = parser.parse_ssa_value()
        return cls.build(result_types=[BoolType([])], operands=[lhs, rhs])

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_ssa_value(self.lhs)
        printer.print(", ")
        printer.print_ssa_value(self.rhs)

    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ:
            raise ValueError("Operands must have the same type")


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
class ConstantBoolOp(Operation, SMTLibOp):
    name = "smt.constant_bool"
    res = ResultDef(BoolType)
    value = AttributeDef(BoolAttr)

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        if self.value.data:
            print("true", file=stream, end='')
        else:
            print("false", file=stream, end='')

    @staticmethod
    def from_bool(value: bool) -> ConstantBoolOp:
        return ConstantBoolOp.create(result_types=[BoolType([])],
                                     attributes={"value": BoolAttr(value)})

    @classmethod
    def parse(cls, result_types: list[Attribute],
              parser: Parser) -> ConstantBoolOp:
        if parser.parse_optional_string("true"):
            return cls.from_bool(True)
        if parser.parse_optional_string("false"):
            return cls.from_bool(False)
        raise ValueError("Expected 'true' or 'false'")

    def print(self, printer: Printer) -> None:
        if self.value.data:
            printer.print(" true")
        else:
            printer.print(" false")


@irdl_op_definition
class NotOp(Operation, SimpleSMTLibOp):
    name = "smt.not"
    res = ResultDef(BoolType)
    arg = OperandDef(BoolType)

    @classmethod
    def parse(cls, result_types: list[Attribute], parser: Parser) -> NotOp:
        val = parser.parse_ssa_value()
        return cls.build(result_types=[BoolType([])], operands=[val])

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_ssa_value(self.arg)

    def op_name(self) -> str:
        return "not"


@irdl_op_definition
class ImpliesOp(BinaryBoolOp, SimpleSMTLibOp):
    name = "smt.implies"

    def op_name(self) -> str:
        return "=>"


@irdl_op_definition
class AndOp(BinaryBoolOp, SimpleSMTLibOp):
    name = "smt.and"

    def op_name(self) -> str:
        return "and"


@irdl_op_definition
class OrOp(BinaryBoolOp, SimpleSMTLibOp):
    name = "smt.or"

    def op_name(self) -> str:
        return "or"


@irdl_op_definition
class XorOp(BinaryBoolOp, SimpleSMTLibOp):
    name = "smt.xor"

    def op_name(self) -> str:
        return "xor"


@irdl_op_definition
class EqOp(BinaryTOp, SimpleSMTLibOp):
    name = "smt.eq"

    def op_name(self) -> str:
        return "="


@irdl_op_definition
class DiscinctOp(BinaryTOp, SimpleSMTLibOp):
    name = "smt.distinct"

    def op_name(self) -> str:
        return "distinct"


@irdl_op_definition
class IteOp(Operation, SimpleSMTLibOp):
    name = "smt.ite"
    res = ResultDef(Attribute)
    cond = OperandDef(BoolType)
    true_val = OperandDef(Attribute)
    false_val = OperandDef(Attribute)

    @classmethod
    def parse(cls: type[IteOp], result_types: list[Attribute],
              parser: Parser) -> IteOp:
        cond = parser.parse_ssa_value()
        parser.parse_string(",")
        true_val = parser.parse_ssa_value()
        parser.parse_string(",")
        false_val = parser.parse_ssa_value()
        return cls.create(result_types=[true_val.typ],
                          operands=[cond, true_val, false_val])

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_ssa_value(self.cond)
        printer.print(", ")
        printer.print_ssa_value(self.true_val)
        printer.print(", ")
        printer.print_ssa_value(self.false_val)

    def verify_(self) -> None:
        if not (self.res.typ == self.true_val.typ == self.false_val.typ):
            raise ValueError(
                "The result and both values must have the same type")

    def op_name(self) -> str:
        return "ite"


@dataclass
class SMTDialect:
    ctx: MLContext

    def __post_init__(self):
        # Base SMTLib language
        self.ctx.register_attr(BoolType)
        self.ctx.register_op(YieldOp)
        self.ctx.register_op(ForallOp)
        self.ctx.register_op(ExistsOp)
        self.ctx.register_op(CallOp)

        # SMTLib Scripting
        self.ctx.register_op(DefineFunOp)
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
