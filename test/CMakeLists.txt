# SPDX-License-Identifier: Apache-2.0

# The backend tests require ONNX package installation
add_subdirectory(backend)

# The lit and unit tests do NOT require ONNX package installation
add_subdirectory(mlir)
add_subdirectory(numerical)
add_subdirectory(onnx2mlir)
add_subdirectory(unit)
add_subdirectory(compilerlib)
