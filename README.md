# MLIR: SMT Dialect (out-of-tree)

forked from [jmgorius/standalone-template](https://github.com/jmgorius/mlir-standalone-template/)

An MLIR dialect to represent SMTLib expressions, and tools to connect SMT solvers and MLIR `opt`.

## How to build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
ninja check-smt-opt
```
To build the documentation from the TableGen description of the dialect
operations, run
```sh
ninja mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with
CMake so that it installs `FileCheck` to the chosen installation prefix.

## Include as subproject

Add this repo as a submodule (say `smt-dialect`), and add this to your main project's CMakeLists.txt:
```
add_subdirectory(smt-dialect)
```

## License

This dialect template is made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.


## Running the xdsl-smt project

Cd to the right folder

```
cd playground/xdsl-smt
```

Create a python environment

```
python -m venv venv
```

Checkout the environment

```
source venv/bin/activate
```

Install the requirements

```
pip install -r requirements.txt
```

Run xdsl-opt on an example

```
python xdsl-smt.py my_file -p=pass1,pass2 -t smt
```
