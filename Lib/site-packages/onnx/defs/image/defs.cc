/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <functional>

#include "onnx/defs/data_type_utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

static const char* ImageDecoder_ver20_doc =
    R"DOC(Loads and decodes and image from a file. If it can't decode for any reason (e.g. corrupted encoded
stream, invalid format, it will return an empty matrix).
The following image formats are supported:
* BMP
* JPEG (note: Lossless JPEG support is optional)
* JPEG2000
* TIFF
* PNG
* WebP
* Portable image format (PBM, PGM, PPM, PXM, PNM)
Decoded images follow a channel-last layout: (Height, Width, Channels).
**JPEG chroma upsampling method:**
When upsampling the chroma components by a factor of 2, the pixels are linearly interpolated so that the
centers of the output pixels are 1/4 and 3/4 of the way between input pixel centers.
When rounding, 0.5 is rounded down and up at alternative pixels locations to prevent bias towards
larger values (ordered dither pattern).
Considering adjacent input pixels A, B, and C, B is upsampled to pixels B0 and B1 so that
```
B0 = round_half_down((1/4) * A + (3/4) * B)
B1 = round_half_up((3/4) * B + (1/4) * C)
```
This method,  is the default chroma upsampling method in the well-established libjpeg-turbo library,
also referred as "smooth" or "fancy" upsampling.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ImageDecoder,
    20,
    OpSchema()
        .SetDoc(ImageDecoder_ver20_doc)
        .Attr(
            "pixel_format",
            "Pixel format. Can be one of \"RGB\", \"BGR\", or \"Grayscale\".",
            AttributeProto::STRING,
            std::string("RGB"))
        .Input(0, "encoded_stream", "Encoded stream", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "image", "Decoded image", "T2", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint("T1", {"tensor(uint8)"}, "Constrain input types to 8-bit unsigned integer tensor.")
        .TypeConstraint("T2", {"tensor(uint8)"}, "Constrain output types to 8-bit unsigned integer tensor.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = getInputShape(ctx, 0);
            if (input_shape.dim_size() != 1) {
              fail_shape_inference("Input tensor must be 1-dimensional");
            }
          }
          propagateElemTypeFromDtypeToOutput(ctx, TensorProto::UINT8, 0);
          auto output_type = ctx.getOutputType(0);
          auto* sh = output_type->mutable_tensor_type()->mutable_shape();
          sh->clear_dim();
          sh->add_dim();
          sh->add_dim();
          sh->add_dim();
        }));

} // namespace ONNX_NAMESPACE
