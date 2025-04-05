/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

static const char* QuantizeLinear_ver19_doc = R"DOC(
The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
The quantization formula is `y = saturate ((x / y_scale) + y_zero_point)`.
For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
'y_zero_point' and 'y' must have same type.
'y_zero_point' is usually not used for quantization to float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz,
but the quantization formula remains the same for consistency and
the type of the attribute 'y_zero_point' still determines the quantization type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    19,
    OpSchema()
        .Input(0, "x", "N-D full precision Input tensor to be quantized.", "T1")
        .Input(
            1,
            "y_scale",
            "Scale for doing quantization to get 'y'. It can be a scalar, which means per-tensor/layer quantization, "
            "or a 1-D Tensor for per-axis quantization.",
            "T1")
        .Input(
            2,
            "y_zero_point",
            "Zero point for doing quantization to get 'y'. Shape must match y_scale. "
            "Default is uint8 with zero point of 0 if it's not specified.",
            "T2",
            OpSchema::Optional)
        .Output(0, "y", "N-D quantized output tensor. It has same shape as input 'x'.", "T2")
        .Attr(
            "axis",
            "(Optional) The axis of the quantization dimension of the input tensor. Ignored for per-tensor quantization. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "saturate",
            "The parameter defines how the conversion behaves if an input value is out of "
            "range of the destination type. It only applies for float 8 quantization "
            "(float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz). It is true by default. "
            "All cases are fully described in two tables inserted in the operator description.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(int32)"},
            "Constrain 'x' to float, float16, bfloat16 or int32 tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)",
             "tensor(uint8)",
             "tensor(float8e4m3fn)",
             "tensor(float8e4m3fnuz)",
             "tensor(float8e5m2)",
             "tensor(float8e5m2fnuz)"},
            "Constrain 'y_zero_point' and 'y' to 8-bit integer/float tensor.")
        .SetDoc(QuantizeLinear_ver19_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          if (ctx.hasInput(2)) {
            propagateElemTypeFromInputToOutput(ctx, 2, 0);
          } else {
            updateOutputElemType(ctx, 0, TensorProto::UINT8);
          }
          if (!hasInputShape(ctx, 0)) {
            return;
          }

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* DequantizeLinear_ver19_doc = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point` must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
`x_zero_point` and `x` must have same type. `x` and `y` must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz quantization,
but the dequantization formula remains the same for consistency and 'x_scale' still determines the output type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    19,
    OpSchema()
        .Input(0, "x", "N-D quantized input tensor to be de-quantized.", "T1")
        .Input(
            1,
            "x_scale",
            "Scale for input 'x'. It can be a scalar, which means a per-tensor/layer dequantization, "
            "or a 1-D tensor for per-axis dequantization.",
            "T2")
        .Input(
            2,
            "x_zero_point",
            "Zero point for input 'x'. Shape must match x_scale. "
            "It's optional. Zero point is 0 when it's not specified.",
            "T1",
            OpSchema::Optional)
        .Output(0, "y", "N-D full precision output tensor. It has same shape as input 'x'.", "T2")
        .Attr(
            "axis",
            "(Optional) The axis of the dequantizing dimension of the input tensor. Used only for per-axis quantization. "
            "Negative value means counting dimensions from the back. Accepted range is `[-r, r-1]` "
            "where `r = rank(input)`. When the rank of the input is 1, per-tensor quantization is applied, "
            "rendering the axis unnecessary in this scenario.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeConstraint(
            "T1",
            {"tensor(int8)",
             "tensor(uint8)",
             "tensor(int32)",
             "tensor(float8e4m3fn)",
             "tensor(float8e4m3fnuz)",
             "tensor(float8e5m2)",
             "tensor(float8e5m2fnuz)"},
            "Constrain 'x_zero_point' and 'x' to 8-bit integer or float, or /32-bit integer tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
            "'x_scale' determines the output type.")
        .SetDoc(DequantizeLinear_ver19_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* QuantizeLinear_ver13_doc = R"DOC(
The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point).
For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    13,
    OpSchema()
        .Input(0, "x", "N-D full precision Input tensor to be quantized.", "T1")
        .Input(
            1,
            "y_scale",
            "Scale for doing quantization to get 'y'. It can be a scalar, which means per-tensor/layer quantization, "
            "or a 1-D Tensor for per-axis quantization.",
            "tensor(float)")
        .Input(
            2,
            "y_zero_point",
            "Zero point for doing quantization to get 'y'. Shape must match y_scale. "
            "Default is uint8 with zero point of 0 if it's not specified.",
            "T2",
            OpSchema::Optional)
        .Output(0, "y", "N-D quantized output tensor. It has same shape as input 'x'.", "T2")
        .Attr(
            "axis",
            "(Optional) The axis of the quantization dimension of the input tensor. Ignored for per-tensor quantization. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeConstraint("T1", {"tensor(float)", "tensor(int32)"}, "Constrain 'x' to float or int32 tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain 'y_zero_point' and 'y' to 8-bit integer tensor.")
        .SetDoc(QuantizeLinear_ver13_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          if (ctx.hasInput(2)) {
            propagateElemTypeFromInputToOutput(ctx, 2, 0);
          } else {
            updateOutputElemType(ctx, 0, TensorProto::UINT8);
          }
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* DequantizeLinear_ver13_doc = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point` must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
`x_zero_point` and `x` must have same type. `x` and `y` must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    13,
    OpSchema()
        .Input(0, "x", "N-D quantized input tensor to be de-quantized.", "T")
        .Input(
            1,
            "x_scale",
            "Scale for input 'x'. It can be a scalar, which means a per-tensor/layer dequantization, "
            "or a 1-D tensor for per-axis dequantization.",
            "tensor(float)")
        .Input(
            2,
            "x_zero_point",
            "Zero point for input 'x'. Shape must match x_scale. "
            "It's optional. Zero point is 0 when it's not specified.",
            "T",
            OpSchema::Optional)
        .Output(0, "y", "N-D full precision output tensor. It has same shape as input 'x'.", "tensor(float)")
        .Attr(
            "axis",
            "(Optional) The axis of the dequantizing dimension of the input tensor. Ignored for per-tensor quantization. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeConstraint(
            "T",
            {"tensor(int8)", "tensor(uint8)", "tensor(int32)"},
            "Constrain 'x_zero_point' and 'x' to 8-bit/32-bit integer tensor.")
        .SetDoc(DequantizeLinear_ver13_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto y_type = ctx.getOutputType(0);
          // only float is supported
          y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);

          if (!hasInputShape(ctx, 0))
            return;

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* QuantizeLinear_ver10_doc = R"DOC(
The linear per-tensor/layer quantization operator. It consumes a high precision tensor, a scale, a zero point to compute the low precision / quantized tensor.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point). For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    10,
    OpSchema()
        .Input(0, "x", "N-D full precision Input tensor to be quantized.", "T1")
        .Input(
            1,
            "y_scale",
            "Scale for doing quantization to get 'y'. It's a scalar, which means a per-tensor/layer quantization.",
            "tensor(float)")
        .Input(
            2,
            "y_zero_point",
            "Zero point for doing quantization to get 'y'. It's a scalar, which means a per-tensor/layer quantization. "
            "Default value is uint8 typed 0 if it's not specified.",
            "T2",
            OpSchema::Optional)
        .Output(0, "y", "N-D quantized output tensor. It has same shape as input 'x'.", "T2")
        .TypeConstraint("T1", {"tensor(float)", "tensor(int32)"}, "Constrain 'x' to float or int32 tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain 'y_zero_point' and 'y' to 8-bit integer tensor.")
        .SetDoc(QuantizeLinear_ver10_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          if (ctx.hasInput(2)) {
            propagateElemTypeFromInputToOutput(ctx, 2, 0);
          } else {
            updateOutputElemType(ctx, 0, TensorProto::UINT8);
          }
          if (!hasInputShape(ctx, 0)) {
            return;
          }

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* DequantizeLinear_ver10_doc = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, a zero point to compute the full precision tensor.
The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' are both scalars.
'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    10,
    OpSchema()
        .Input(0, "x", "N-D quantized input tensor to be de-quantized.", "T")
        .Input(
            1,
            "x_scale",
            "Scale for input 'x'. It's a scalar, which means a per-tensor/layer quantization.",
            "tensor(float)")
        .Input(
            2,
            "x_zero_point",
            "Zero point for input 'x'. It's a scalar, which means a per-tensor/layer quantization. "
            "It's optional. 0 is the default value when it's not specified.",
            "T",
            OpSchema::Optional)
        .Output(0, "y", "N-D full precision output tensor. It has same shape as input 'x'.", "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(int8)", "tensor(uint8)", "tensor(int32)"},
            "Constrain 'x_zero_point' and 'x' to 8-bit/32-bit integer tensor.")
        .SetDoc(DequantizeLinear_ver10_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto y_type = ctx.getOutputType(0);
          // only float is supported
          y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);

          if (!hasInputShape(ctx, 0))
            return;

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

} // namespace ONNX_NAMESPACE
