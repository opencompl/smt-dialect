// RUN: smt-opt %s | smt-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = smt.foo %{{.*}} : i32
        %res = smt.foo %0 : i32
        return
    }
}
