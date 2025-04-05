/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/math/utils.h"

#include <string>

namespace ONNX_NAMESPACE {
namespace defs {
namespace math {
namespace utils {

int MathOpTwoIntegers(std::string op_type, int a, int b) {
  if (op_type == "Add") {
    return a + b;
  } else if (op_type == "Sub") {
    return a - b;
  } else if (op_type == "Mul") {
    return a * b;
  }
  fail_shape_inference("Wrong op_type name for running propagation: ", op_type);
}

void MatMulShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, int input1Idx, int input2Idx) {
  if (!hasInputShape(ctx, input1Idx) || !hasInputShape(ctx, input2Idx)) {
    return;
  }

  const auto shape0 = ctx.getInputType(input1Idx)->tensor_type().shape();
  const auto shape1 = ctx.getInputType(input2Idx)->tensor_type().shape();

  if (shape0.dim_size() == 0 || shape1.dim_size() == 0) {
    fail_shape_inference("Input tensors of wrong rank (0).");
  }

  ONNX_NAMESPACE::TensorShapeProto shapeL, shapeR;

  // First promote each shape to at least rank-2. This logic is
  // specific to matmul, not generic broadcasting.
  {
    if (shape0.dim_size() == 1) {
      shapeL.add_dim()->set_dim_value(1);
      *shapeL.add_dim() = shape0.dim(0);
    } else {
      *shapeL.mutable_dim() = shape0.dim();
    }
    if (shape1.dim_size() == 1) {
      *shapeR.add_dim() = shape1.dim(0);
      shapeR.add_dim()->set_dim_value(1);
    } else {
      *shapeR.mutable_dim() = shape1.dim();
    }
  }

  // Check for compatible matrix multiply dimensions
  {
    auto dimL = shapeL.dim(shapeL.dim_size() - 1);
    auto dimR = shapeR.dim(shapeR.dim_size() - 2);
    if (dimL.has_dim_value() && dimR.has_dim_value() && dimL.dim_value() != dimR.dim_value()) {
      fail_shape_inference("Incompatible dimensions for matrix multiplication");
    }
  }

  ONNX_NAMESPACE::TensorShapeProto resultShape;

  // Now call out to generic multidimensional broadcasting for
  // the broadcastable prefixes.
  {
    ONNX_NAMESPACE::TensorShapeProto prefixShapeL, prefixShapeR;
    for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
      *prefixShapeL.add_dim() = shapeL.dim(i);
    }
    for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
      *prefixShapeR.add_dim() = shapeR.dim(i);
    }
    bidirectionalBroadcastShapeInference(prefixShapeL, prefixShapeR, resultShape);
  }

  // Back to matmul-specific. Add the trailing dimensions back in.
  {
    if (shape0.dim_size() != 1) {
      *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
    }
    if (shape1.dim_size() != 1) {
      *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
    }
  }

  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = resultShape;
}

void QLinearMatMulShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  auto a_type = ctx.getInputType(0);
  auto b_type = ctx.getInputType(3);
  if (nullptr == a_type || nullptr == b_type || a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
      b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
    fail_type_inference("inputs are expected to have tensor type.");
  }

  auto a_zero_point_type = ctx.getInputType(2);
  if (nullptr == a_zero_point_type ||
      a_zero_point_type->tensor_type().elem_type() != a_type->tensor_type().elem_type()) {
    fail_type_inference("input and zero_point pair is expected to have be same type.");
  }

  auto b_zero_point_type = ctx.getInputType(5);
  if (nullptr == b_zero_point_type ||
      b_zero_point_type->tensor_type().elem_type() != b_type->tensor_type().elem_type()) {
    fail_type_inference("input and zero_point pair is expected to have same type.");
  }

  propagateElemTypeFromInputToOutput(ctx, 7, 0);

  MatMulShapeInference(ctx, 0, 3);
}

const char* QLinearMatMulDoc() {
  static const char* QLinearMatMul_doc = R"DOC(
Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
It consumes two quantized input tensors, their scales and zero points, scale and zero point of output,
and computes the quantized output. The quantization formula is y = saturate((x / y_scale) + y_zero_point).
For (x / y_scale), it is rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or N-D tensor
(per row for 'a' and per column for 'b'). Scalar refers to per tensor quantization whereas N-D refers to per row
or per column quantization. If the input is 2D of shape [M, K] then zero point and scale tensor may be
an M element vector [v_1, v_2, ..., v_M] for per row quantization and K element vector of shape [v_1, v_2, ..., v_K]
for per column quantization. If the input is N-D tensor with shape [D1, D2, M, K] then zero point and scale tensor may
have shape [D1, D2, M, 1] for per row quantization and shape [D1, D2, 1, K] for per column quantization.
Production must never overflow, and accumulation may overflow if and only if in 32 bits.
)DOC";
  return QLinearMatMul_doc;
}

} // namespace utils
} // namespace math
} // namespace defs
} // namespace ONNX_NAMESPACE
