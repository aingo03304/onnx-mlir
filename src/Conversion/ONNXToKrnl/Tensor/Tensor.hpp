/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====- Tensor.hpp - ONNX dialects to Krnl lowering for tensor operations -===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common code shared by ONNX to KRNL conversion operators.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/ONNX/IndexExpr.hpp"

using namespace mlir;

class ONNXShapeConversions : public ConversionPattern {
public:
  ONNXShapeConversions(
      llvm::StringRef opName, PatternBenefit benefit, MLIRContext *ctx)
      : ConversionPattern(opName, benefit, ctx) {}

protected:
  // Reshape the 'input' tensor to the shape prodided by 'outputDims'.
  Value reshape(const Value input, const ArrayRef<DimIndexExpr> outputDims,
      ConversionPatternRewriter &rewriter, const Location &loc) const;

  // Transpose the 'input' tensor given the permutation array.
  Value transpose(const Value input, const ArrayRef<DimIndexExpr> outputDims,
      const ArrayRef<int64_t> perm, ConversionPatternRewriter &rewriter,
      const Location &loc) const;
};
