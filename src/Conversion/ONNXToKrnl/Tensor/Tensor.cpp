/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====- Tensor.cpp - ONNX dialects to Krnl lowering for tensor operations -===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common code shared by ONNX to KRNL conversion operators.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/Tensor/Tensor.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

// Reshape the 'input' tensor to the shape prodided by 'outputDims'.
Value ONNXShapeConversions::reshape(const Value input,
    const ArrayRef<DimIndexExpr> outputDims,
    ConversionPatternRewriter &rewriter, const Location &loc) const {
  assert(!outputDims.empty() && "Output dimensions should not be empty");

  OnnxBuilder onnxBuilder(rewriter, loc);

  // If the output dimensions are all literals the 'onnx/Reshape' operation
  // can take the new shape via an 'onnx.Constant'.
  if (llvm::all_of(outputDims,
          [](const DimIndexExpr &dim) { return dim.isLiteral(); })) {
    SmallVector<int64_t, 6> shape;
    for (const IndexExpr &dim : outputDims)
      shape.push_back(dim.getLiteral());

    auto constantOp = getONNXConstantOpFromDenseAttr(
        rewriter, loc, rewriter.getI64TensorAttr(shape));

    ShapedType inputType = input.getType().cast<ShapedType>();
    Type elementType = inputType.getElementType();
    Value reshapeRes = onnxBuilder.reshape(
        MemRefType::get(shape, elementType), input, constantOp);

    return reshapeRes;
  }

  MemRefBuilder memRefBuilder(onnxBuilder);
  KrnlBuilder krnlBuilder(onnxBuilder);

  // When the output dimensions aren't all literals we need to generate code
  // to compute the shape. Allocate a buffer and store the putput dimension
  // into it.
  IndexType indexTy = rewriter.getIndexType();
  int64_t length = outputDims.size();
  memref::AllocOp alloc =
      memRefBuilder.alignedAlloc(MemRefType::get({length}, indexTy), 16);

  for (int64_t i = 0; i < length; ++i) {
    Value index = emitConstantOp(rewriter, loc, indexTy, i);
    Value data = outputDims[i].getValue();
    krnlBuilder.store(data, alloc, index);
  }

  // Now create the "onnx.Reshape" operation. Because the shape is not a
  // compile time constant it is effectively unknown.
  SmallVector<int64_t> shape(length, -1);
  ShapedType inputType = input.getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();
  Value reshapeRes =
      onnxBuilder.reshape(MemRefType::get(shape, elementType), input, alloc);

  // The 'onnx.Reshape' operation yields a memref with unknown extents, so we
  // need to explicitly cast the result to the know size.
  SmallVector<int64_t, 6> castOutputShape;
  for (const IndexExpr &dim : outputDims)
    castOutputShape.push_back(dim.isLiteral() ? dim.getLiteral() : -1);

  Value castRes = memRefBuilder.cast(
      reshapeRes, MemRefType::get(castOutputShape, elementType));

  return castRes;
}

// Transpose the 'input' tensor given the permutation array.
Value ONNXShapeConversions::transpose(const Value input,
    const ArrayRef<DimIndexExpr> outputDims, const ArrayRef<int64_t> perm,
    ConversionPatternRewriter &rewriter, const Location &loc) const {
  assert(!outputDims.empty() && "Output dimensions should not be empty");
  assert(!perm.empty() && perm.size() == outputDims.size() &&
         "Expecitng valid permutation array");

  // Compute the shape of the 'onnx.Transpose' result.
  SmallVector<int64_t, 6> shape;
  for (const IndexExpr &dim : outputDims)
    shape.push_back(dim.isLiteral() ? dim.getLiteral() : -1);

  // Compute the memref type of the "onnx.Transpose" output.
  ShapedType inputType = input.getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();
  MemRefType memRefType = MemRefType::get(shape, elementType);

  // Create the "onnx.Transpose" operation.
  OnnxBuilder onnxBuilder(rewriter, loc);
  Value transposeRes =
      onnxBuilder.transpose(memRefType, input, rewriter.getI64ArrayAttr(perm));

  return transposeRes;
}
