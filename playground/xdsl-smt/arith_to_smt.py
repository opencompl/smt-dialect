from dataclasses import dataclass, field
from typing import Pattern, cast
from xdsl.dialects.builtin import FunctionType, ModuleOp, IntegerType
from xdsl.dialects.func import FuncOp, Return
from xdsl.ir import (Attribute, MLContext, Operation, SSAValue, Block, Region)
from xdsl.pattern_rewriter import (PatternRewriteWalker, PatternRewriter,
                                   RewritePattern, op_type_rewrite_pattern)

from bitvector_dialect import BitVectorType, ConstantOp
from dialect import BoolType, DefineFunOp, ReturnOp
from utils_dialect import PairType


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


def convert_return(op: Return, ssa_mapping: dict[SSAValue,
                                                 SSAValue]) -> list[Operation]:
    if len(op.arguments) != 1:
        raise Exception("Return must have one argument")
    return [ReturnOp.create(operands=[ssa_mapping[op.arguments[0]]])]


def convert_operation(
        op: Operation, ssa_mapping: dict[SSAValue,
                                         SSAValue]) -> list[Operation]:
    if isinstance(op, Return):
        return convert_return(op, ssa_mapping)
    raise Exception("Cannot convert ", type(op), " operation")


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
        smt_func_res_type = [convert_type(res) for res in func_res_type]

        # Create the function
        smt_func = DefineFunOp.from_function_type(
            FunctionType.from_lists(smt_func_operands_type, smt_func_res_type))

        # Map the new ssa values
        for i, arg in enumerate(smt_func.body.blocks[0].args):
            ssa_mapping[func_operands[i]] = arg

        new_body_ops = list[Operation]()
        for body_op in op.body.blocks[0].ops:
            new_body_ops.extend(convert_operation(body_op, ssa_mapping))
        rewriter.insert_op_at_pos(new_body_ops, smt_func.body.blocks[0], 0)

        # Convert the function body
        rewriter.replace_matched_op(smt_func, new_results=[])


def arith_to_smt(ctx: MLContext, module: ModuleOp) -> None:
    walker = PatternRewriteWalker(FuncToSMTPattern())
    walker.rewrite_module(module)
