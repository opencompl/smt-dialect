#ifndef TARGET_MLIRTOSMT_H_
#define TARGET_MLIRTOSMT_H_

namespace mlir {
void registerMLIRToSMTTranslation(
    std::function<void(DialectRegistry &)> registrationCallback =
        [](DialectRegistry &) {});
} // namespace mlir

#endif // TARGET_MLIRTOSMT_H_