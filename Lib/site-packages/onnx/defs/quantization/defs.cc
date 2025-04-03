/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

static const char* QuantizeLinear_ver21_doc = R"DOC(
The linear quantization operator consumes a high-precision tensor, a scale, and a zero point to compute the
low-precision/quantized tensor. The scale factor and zero point must have the same shape, determining the quantization
granularity. The quantization formula is `y = saturate((x / y_scale) + y_zero_point)`.

Saturation is done according to:
- uint16: [0, 65535]
- int16: [-32768, 32767]
- uint8: [0, 255]
- int8: [-128, 127]
- uint4: [0, 15]
- int4: [-8, 7]

For `(x / y_scale)`, it rounds to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.

`y_zero_point` and `y` must have the same type. `y_zero_point` is usually not used for quantization to float8 types, but the quantization
formula remains the same for consistency, and the type of the attribute `y_zero_point` still determines the quantization type.

There are three supported quantization granularities, determined by the shape of `y_scale`.
In all cases, `y_zero_point` must have the same shape as `y_scale`.
- Per-tensor (per-layer) quantization: `y_scale` is a scalar.
- Per-axis quantization: The scale must be a 1-D tensor, with the length of the quantization axis. For an input shape
 `(D0, ..., Di, ..., Dn)` and `axis=i`, `y_scale` is a 1-D tensor of length `Di`.
- Blocked quantization: The scale's shape is identical to the input's shape, except for one dimension, in which
  blocking is performed. Given `x` shape `(D0, ..., Di, ..., Dn)`, `axis=i`, and block size `B`: `y_scale` shape is
  `(D0, ..., ceil(Di/B), ..., Dn)`.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    21,
    OpSchema()
        .Input(0, "x", "N-D full precision Input tensor to be quantized.", "T1")
        .Input(
            1,
            "y_scale",
            "Scale for doing quantization to get `y`. For per-tensor/layer quantization the scale is a scalar, for "
            "per-axis quantization it is a 1-D Tensor and for blocked quantization it has the same shape as the "
            "input, except for one dimension in which blocking is performed.",
            "T1")
        .Input(
            2,
            "y_zero_point",
            "Zero point for doing quantization to get `y`. Shape must match `y_scale`."
            "Default is uint8 with zero point of 0 if it's not specified.",
            "T2",
            OpSchema::Optional)
        .Output(0, "y", "N-D quantized output tensor. It has same shape as input `x`.", "T2")
        .Attr(
            "axis",
            "(Optional) The axis of the dequantizing dimension of the input tensor. Used only for per-axis and blocked "
            "quantization. Negative value means counting dimensions from the back. Accepted range is `[-r, r-1]` "
            "where `r = rank(input)`. When the rank of the input is 1, per-tensor quantization is applied, "
            "rendering the axis unnecessary in this scenario.",
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
        .Attr(
            "block_size",
            "(Optional) The size of the quantization block (number of times every scale is replicated). Used only for "
            "blocked quantization. The block size is a positive integer. Given `x` shape `(D0, ..., Di, ..., Dn)`, "
            "`y_scale` shape `(S0, ... Si, ...Sn)` and `axis=i`, the accepted range is "
            "`[ceil(Di/Si), ceil(Di/(Si-1))-1]`",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "output_dtype",
            "(Optional) The output data type. If not supplied, the output data type is inferred from `y_zero_point` data type (`T2`). "
            "If neither `output_dtype` nor `y_zero_point` are supplied, output data type is uint8. "
            "If both `output_dtype` and `y_zero_point` are specified, `output_dtype` must be `T2`.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(int32)"},
            "The type of the input 'x'.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)",
             "tensor(uint8)",
             "tensor(int16)",
             "tensor(uint16)",
             "tensor(float8e4m3fn)",
             "tensor(float8e4m3fnuz)",
             "tensor(float8e5m2)",
             "tensor(float8e5m2fnuz)",
             "tensor(uint4)",
             "tensor(int4)"},
            "The type of the input `y_zero_point` and the output `y`.")
        .SetDoc(QuantizeLinear_ver21_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto const zp_type = ctx.hasInput(2) ? ctx.getInputType(2) : nullptr;
          auto const output_dtype =
              static_cast<TensorProto_DataType>(getAttribute(ctx, "output_dtype", TensorProto::UNDEFINED));
          if (zp_type != nullptr) {
            auto const zp_elem_type = static_cast<TensorProto_DataType>(getTensorElementType(*zp_type));
            if (output_dtype != TensorProto::UNDEFINED && output_dtype != zp_elem_type) {
              fail_type_inference(
                  "output_dtype ",
                  TensorProto_DataType_Name(output_dtype),
                  " does not match y_zero_point type ",
                  TensorProto_DataType_Name(zp_elem_type),
                  ".");
            }
            propagateElemTypeFromInputToOutput(ctx, 2, 0);
          } else if (output_dtype != TensorProto::UNDEFINED) {
            propagateElemTypeFromAttributeToOutput(ctx, "output_dtype", 0);
          } else {
            updateOutputElemType(ctx, 0, TensorProto::UINT8);
          }
          if (!hasInputShape(ctx, 0)) {
            return;
          }

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* DequantizeLinear_ver21_doc = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the
full-precision tensor. The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point`
must have the same shape, determining the quantization's granularity: a scalar for per-tensor/per-layer quantization,
a 1-D tensor for per-axis quantization, or have a rank identical to the input for blocked quantization.
See QuantizeLinear for details on quantization granularity.

`x_zero_point` and `x` must have the same type. `x` and `y` must have the same shape. In the case of dequantizing
`int32`, there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8 types quantization, but the dequantization formula remains the same
for consistency, and `x_scale` still determines the output type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    21,
    OpSchema()
        .Input(0, "x", "N-D quantized input tensor to be de-quantized.", "T1")
        .Input(
            1,
            "x_scale",
            "Scale for input `x`. For per-tensor/layer dequantization the scale is a scalar, for "
            "per per-axis dequantization it is a 1-D Tensor and for blocked dequantization it has the same shape as "
            "the input, except for one dimension in which blocking is performed.",
            "T2")
        .Input(
            2,
            "x_zero_point",
            "Zero point for input `x`. Shape must match x_scale. "
            "It's optional. Zero point is 0 when it's not specified.",
            "T1",
            OpSchema::Optional)
        .Output(0, "y", "N-D full precision output tensor. It has same shape as input `x`.", "T2")
        .Attr(
            "axis",
            "(Optional) The axis of the dequantizing dimension of the input tensor. Used for per-axis and blocked "
            "quantization. Negative value means counting dimensions from the back. Accepted range is `[-r, r-1]` "
            "where `r = rank(input)`.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "block_size",
            "(Optional) The size of the quantization block (number of times every scale is replicated). Used only for "
            "blocked quantization. The block size is a positive integer. Given `x` shape `(D0, ..., Di, ..., Dn)`, "
            "`y_scale` shape `(S0, ... Si, ...Sn)` and `axis=i`, the accepted range is "
            "`[ceil(Di/Si), ceil(Di/(Si-1))-1]`",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .TypeConstraint(
            "T1",
            {"tensor(int8)",
             "tensor(uint8)",
             "tensor(int16)",
             "tensor(uint16)",
             "tensor(int32)",
             "tensor(float8e4m3fn)",
             "tensor(float8e4m3fnuz)",
             "tensor(float8e5m2)",
             "tensor(float8e5m2fnuz)",
             "tensor(uint4)",
             "tensor(int4)"},
            "The type of the inputs 'x_zero_point' and 'x'.")
        .TypeConstraint(
            "T2",
            {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
            "'x_scale' determines the output type.")
        .SetDoc(DequantizeLinear_ver21_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* DynamicQuantizeLinear_ver11_doc = R"DOC(
A Function to fuse calculation for Scale, Zero Point and FP32->8Bit conversion of FP32 Input data.
Outputs Scale, ZeroPoint and Quantized Input for a given FP32 Input.
Scale is calculated as:
```
y_scale = (maximum(0, max(x)) - minimum(0, min(x))) / (qmax - qmin)
```

* where qmax and qmin are max and min values for quantization range i.e. [0, 255] in case of uint8
* data range is adjusted to include 0.

Zero point is calculated as:
```
intermediate_zero_point = qmin - min(x)/y_scale
y_zero_point = cast(round(saturate(itermediate_zero_point)))
```

* where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.

Data quantization formula is:
```
y = saturate (round (x / y_scale) + y_zero_point)
```

* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DynamicQuantizeLinear,
    11,
    OpSchema()
        .SetDoc(DynamicQuantizeLinear_ver11_doc)
        .Input(0, "x", "Input tensor", "T1")
        .Output(0, "y", "Quantized output tensor", "T2")
        .Output(
            1,
            "y_scale",
            "Output scale. It's a scalar, which means a per-tensor/layer quantization.",
            "tensor(float)")
        .Output(
            2,
            "y_zero_point",
            "Output zero point. It's a scalar, which means a per-tensor/layer quantization.",
            "T2")
        .TypeConstraint("T1", {"tensor(float)"}, "Constrain 'x' to float tensor.")
        .TypeConstraint("T2", {"tensor(uint8)"}, "Constrain 'y_zero_point' and 'y' to 8-bit unsigned integer tensor.")
        .FunctionBody(R"ONNX(
        {
           Q_Min = Constant<value = float {0.0}>()
           Q_Max = Constant<value = float {255.0}>()
           X_Min = ReduceMin <keepdims = 0> (x)
           X_Min_Adjusted = Min (X_Min, Q_Min)
           X_Max = ReduceMax <keepdims = 0> (x)
           X_Max_Adjusted = Max (X_Max, Q_Min)
           X_Range = Sub (X_Max_Adjusted, X_Min_Adjusted)
           Scale = Div (X_Range, Q_Max)
           Min_Scaled = Div (X_Min_Adjusted, Scale)
           Initial_ZeroPoint_FP = Sub (Q_Min, Min_Scaled)
           Clipped_ZeroPoint_FP = Clip (Initial_ZeroPoint_FP, Q_Min, Q_Max)
           Rounded_ZeroPoint_FP = Round (Clipped_ZeroPoint_FP)
           Zeropoint = Cast <to = 2> (Rounded_ZeroPoint_FP)
           y_scale = Identity (Scale)
           y_zero_point = Identity (Zeropoint)
           y = QuantizeLinear (x, Scale, Zeropoint)
        }
        )ONNX")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          updateOutputElemType(ctx, 0, TensorProto::UINT8);
          updateOutputElemType(ctx, 1, TensorProto::FLOAT);
          updateOutputElemType(ctx, 2, TensorProto::UINT8);

          ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
          ctx.getOutputType(2)->mutable_tensor_type()->mutable_shape();

          if (!hasInputShape(ctx, 0))
            return;

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

} // namespace ONNX_NAMESPACE
