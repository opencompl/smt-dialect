//===- SMTDialect.td - SMT dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SMT_DIALECT
#define SMT_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SMT dialect definition.
//===----------------------------------------------------------------------===//

def SMT_Dialect : Dialect {
    let name = "smt";
    let summary = "MLIR SMT dialect";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::smt";
}

//===----------------------------------------------------------------------===//
// Base SMT operation definition.
//===----------------------------------------------------------------------===//

class SMT_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<SMT_Dialect, mnemonic, traits>;

#endif // SMT_DIALECT
