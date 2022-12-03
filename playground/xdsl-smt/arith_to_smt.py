from dataclasses import dataclass, field
from typing import Pattern, cast
from xdsl.dialects.builtin import AnyIntegerAttr, FunctionType, IntegerAttr, ModuleOp, IntegerType
from xdsl.dialects.func import FuncOp, Return
from xdsl.ir import (Attribute, MLContext, Operation, SSAValue, Block, Region)
from xdsl.pattern_rewriter import (PatternRewriteWalker, PatternRewriter,
                                   RewritePattern, op_type_rewrite_pattern)
from arith_dialect import Constant

from bitvector_dialect import BitVectorType, ConstantOp
from dialect import BoolType, ConstantBoolOp, DefineFunOp, ReturnOp
from utils_dialect import PairOp, PairType
import bitvector_dialect as bv_dialect


def int_smt_type(bitwidth: int) -> PairType[BitVectorType, BoolType]:
    """
    The SMT type of an integer. It contains the integer value, and a boolean
    indicating if the value is poison.
    We could also rewrite this to be a sum type rather than a product.
    """
    return PairType.from_params(BitVectorType.from_int(bitwidth), BoolType())


def convert_type(attr: Attribute) -> Attribute:
    """Convert an arith type to the corresponding SMT type."""
    if not isinstance(attr, IntegerType):
        raise Exception("Cannot convert ", type(attr), " type")
    return int_smt_type(attr.width.data)


def get_constant_bv_ops(value: int, width: int) -> list[Operation]:
    """
    Create operations returning a bitvector constant,
    as well as the poison indicator.
    """
    constant = bv_dialect.ConstantOp.from_int_value(value, width)
    poison = ConstantBoolOp.from_bool(False)
    pair = PairOp.from_values(constant.res, poison.res)
    return [constant, poison, pair]


def convert_return(op: Return, ssa_mapping: dict[SSAValue, SSAValue],
                   ub_value: SSAValue) -> list[Operation]:
    if len(op.arguments) != 1:
        raise Exception("Return must have one argument")
    create_pair = PairOp.from_values(ub_value, ssa_mapping[op.arguments[0]])
    return_ = ReturnOp.from_ret_values([create_pair.res])
    return [create_pair, return_]


def convert_constant(op: Constant,
                     ssa_mapping: dict[SSAValue, SSAValue]) -> list[Operation]:
    constant_value = cast(IntegerAttr[IntegerType], op.value)
    return get_constant_bv_ops(constant_value.value.data,
                               constant_value.typ.width.data)


def convert_operation(op: Operation, ssa_mapping: dict[SSAValue, SSAValue],
                      ub_value: SSAValue) -> tuple[list[Operation], SSAValue]:
    if isinstance(op, Return):
        new_ops = convert_return(op, ssa_mapping, ub_value)
    elif isinstance(op, Constant):
        new_ops = convert_constant(op, ssa_mapping)
    else:
        raise Exception("Cannot convert ", type(op), " operation")
    for index, res in enumerate(op.results):
        ssa_mapping[res] = new_ops[-1].results[index]
    return new_ops, ub_value


class FuncToSMTPattern(RewritePattern):

    def match_and_rewrite(self, op: Operation,
                          rewriter: PatternRewriter) -> None:
        if not isinstance(op, FuncOp):
            return None
        ssa_mapping = dict[SSAValue, SSAValue]()

        # Get the function input types
        func_operands = op.body.blocks[0].args
        func_operands_type = [arg.typ for arg in func_operands]
        smt_func_operands_type = [
            convert_type(arg) for arg in func_operands_type
        ]

        # Get the function output types
        func_res_type: list[Attribute] = op.function_type.outputs.data
        if len(func_res_type) != 1:
            raise Exception(
                "arith-to-smt: func.fnc must have exactly one result")
        smt_func_res_type = [
            PairType.from_params(BoolType(), convert_type(func_res_type[0]))
        ]

        # Create the function
        smt_func = DefineFunOp.from_function_type(FunctionType.from_lists(
            smt_func_operands_type, smt_func_res_type),
                                                  name=op.sym_name)

        # Map the new ssa values
        for i, arg in enumerate(smt_func.body.blocks[0].args):
            ssa_mapping[func_operands[i]] = arg

        # Add false to the UB marker
        new_body_ops = list[Operation]()
        init_ub = ConstantBoolOp.from_bool(False)
        new_body_ops.append(init_ub)
        ub_value = init_ub.res

        # Convert the function body
        for body_op in op.body.blocks[0].ops:
            new_ops, ub_value = convert_operation(body_op, ssa_mapping,
                                                  ub_value)
            new_body_ops.extend(new_ops)
        rewriter.insert_op_at_pos(new_body_ops, smt_func.body.blocks[0], 0)

        # Convert the function body
        rewriter.replace_matched_op(smt_func, new_results=[])


def arith_to_smt(ctx: MLContext, module: ModuleOp) -> None:
    walker = PatternRewriteWalker(FuncToSMTPattern())
    walker.rewrite_module(module)
