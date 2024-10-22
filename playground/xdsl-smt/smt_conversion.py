from dataclasses import field
from io import IOBase
from dialect import *
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import OpResult, Region, SSAValue


@dataclass
class SMTConversionCtx:
    value_to_name: dict[SSAValue, str] = field(default_factory=dict)
    names: set[str] = field(default_factory=set)

    def get_fresh_name(self, value: str | SSAValue | None) -> str:
        base_name: str
        if isinstance(value, str):
            base_name = value
        elif isinstance(value, SSAValue) and value.name is not None:
            base_name = value.name
        else:
            base_name = "tmp"

        name: str
        if base_name not in self.names:
            name = base_name
        else:
            i = 0
            while f"{base_name}_{i}" in self.names:
                i += 1
            name = f"{base_name}_{i}"

        if isinstance(value, SSAValue):
            self.value_to_name[value] = name
        self.names.add(name)
        return name

    def print_expr_to_smtlib(self, val: SSAValue, stream: IOBase):
        if val in self.value_to_name.keys():
            print(self.value_to_name[val], file=stream, end='')
            return
        assert isinstance(val, OpResult)
        op = val.op
        assert isinstance(op, SMTLibOp)
        op.print_expr_to_smtlib(stream, self)


def print_to_smtlib(module: ModuleOp, stream: IOBase):
    ctx = SMTConversionCtx()
    print("(declare-datatypes ((Pair 2)) "
          "((par (X Y) ((pair (first X) (second Y))))))")
    for op in module.ops:
        if isinstance(op, SMTLibScriptOp):
            op.print_expr_to_smtlib(stream, ctx)
            continue
