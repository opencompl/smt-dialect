#!/usr/bin/env python3

import argparse
from xdsl.xdsl_opt_main import xDSLOptMain
from bitvector_dialect import SMTBitVectorDialect

from dialect import SMTDialect
from bitvector_dialect import SMTBitVectorDialect
from utils_dialect import SMTUtilsDialect
from smt_conversion import print_to_smtlib
from xdsl.dialects.builtin import Builtin
from arith_dialect import Arith

from arith_to_smt import arith_to_smt


class OptMain(xDSLOptMain):

    def register_all_dialects(self):
        Arith(self.ctx)
        Builtin(self.ctx)
        SMTDialect(self.ctx)
        SMTBitVectorDialect(self.ctx)
        SMTUtilsDialect(self.ctx)

    def register_all_passes(self):
        super().register_all_passes()
        self.available_passes["arith_to_smt"] = arith_to_smt

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        super().register_all_arguments(arg_parser)

    def register_all_targets(self):
        super().register_all_targets()
        self.available_targets['smt'] = print_to_smtlib


def __main__():
    xdsl_main = OptMain()
    xdsl_main.run()


if __name__ == "__main__":
    __main__()