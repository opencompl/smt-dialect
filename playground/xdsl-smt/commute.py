from xdsl.ir import Operation, MLContext, Region, Block
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp
from arith_dialect import Addi
from xdsl.rewriter import Rewriter
from xdsl.pattern_rewriter import RewritePattern, PatternRewriter, PatternRewriteWalker

from arith_to_smt import arith_to_smt
from dialect import AssertOp, BoolType, CallOp, CheckSatOp, DeclareConstOp, DefineFunOp, EqOp, NotOp


class CommuteAddiPattern(RewritePattern):

    def match_and_rewrite(self, op: Operation,
                          rewriter: PatternRewriter) -> None:
        if not isinstance(op, Addi):
            return None

        new_op = Addi.create(result_types=[op.operands[0].typ],
                             operands=[op.operands[1], op.operands[0]])
        rewriter.replace_op(op, new_op)


def commute_addi(ctx: MLContext, module: ModuleOp) -> None:
    walker = PatternRewriteWalker(CommuteAddiPattern(),
                                  apply_recursively=False)
    walker.rewrite_module(module)


def commute_tv(ctx: MLContext, module: ModuleOp) -> None:
    new_module = module.clone()
    commute_addi(ctx, new_module)

    Rewriter.inline_block_after(new_module.regions[0].blocks[0],
                                module.ops[-1])

    arith_to_smt(ctx, module)

    funcs = []

    def add_func(op: Operation):
        if isinstance(op, DefineFunOp):
            funcs.append(op)

    module.walk(add_func)

    pre_func = funcs[0]
    post_func = funcs[1]

    num_inputs = len(pre_func.body.blocks[0].args)

    assert num_inputs > 1

    new_ops = []
    vars_ = []
    for input_ in pre_func.body.blocks[0].args:
        var_op = DeclareConstOp.create(result_types=[input_.typ])
        vars_.append(var_op.res)
        new_ops.append(var_op)

    val_pre = CallOp.create(result_types=[pre_func.func_type.outputs.data[0]],
                            operands=[pre_func.ret] + vars_)
    new_ops.append(val_pre)

    val_post = CallOp.create(
        result_types=[post_func.func_type.outputs.data[0]],
        operands=[post_func.ret] + vars_)
    new_ops.append(val_post)

    eq_op = EqOp.create(result_types=[BoolType([])],
                        operands=[val_pre.res, val_post.res])
    new_ops.append(eq_op)

    not_op = NotOp.create(result_types=[BoolType([])], operands=[eq_op.res])
    new_ops.append(not_op)

    assert_ = AssertOp.create(result_types=[], operands=[not_op.res])
    new_ops.append(assert_)

    new_ops.append(CheckSatOp.create())

    module.body.blocks[0].add_ops(new_ops)