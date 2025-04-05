// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_IMAGE_OPS_INTERNAL_H_
#define TENSORFLOW_CC_OPS_IMAGE_OPS_INTERNAL_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {
namespace internal {
// NOTE: This namespace has internal TensorFlow details that
// are not part of TensorFlow's public API.

/// @defgroup image_ops_internal Image Ops Internal
/// @{

/// Extracts a glimpse from the input tensor.
///
/// Returns a set of windows called glimpses extracted at location
/// `offsets` from the input tensor. If the windows only partially
/// overlaps the inputs, the non overlapping areas will be filled with
/// random noise.
///
/// The result is a 4-D tensor of shape `[batch_size, glimpse_height,
/// glimpse_width, channels]`. The channels and batch dimensions are the
/// same as that of the input tensor. The height and width of the output
/// windows are specified in the `size` parameter.
///
/// The argument `normalized` and `centered` controls how the windows are built:
///
/// * If the coordinates are normalized but not centered, 0.0 and 1.0
///   correspond to the minimum and maximum of each height and width
///   dimension.
/// * If the coordinates are both normalized and centered, they range from
///   -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper
///   left corner, the lower right corner is located at (1.0, 1.0) and the
///   center is at (0, 0).
/// * If the coordinates are not normalized they are interpreted as
///   numbers of pixels.
///
/// Args:
/// * scope: A Scope object
/// * input: A 4-D float tensor of shape `[batch_size, height, width, channels]`.
/// * size: A 1-D tensor of 2 elements containing the size of the glimpses
/// to extract.  The glimpse height must be specified first, following
/// by the glimpse width.
/// * offsets: A 2-D integer tensor of shape `[batch_size, 2]` containing
/// the y, x locations of the center of each window.
///
/// Optional attributes (see `Attrs`):
/// * centered: indicates if the offset coordinates are centered relative to
/// the image, in which case the (0, 0) offset is relative to the center
/// of the input images. If false, the (0,0) offset corresponds to the
/// upper left corner of the input images.
/// * normalized: indicates if the offset coordinates are normalized.
/// * uniform_noise: indicates if the noise should be generated using a
/// uniform distribution or a Gaussian distribution.
/// * noise: indicates if the noise should `uniform`, `gaussian`, or
/// `zero`. The default is `uniform` which means the noise type
/// will be decided by `uniform_noise`.
///
/// Returns:
/// * `Output`: A tensor representing the glimpses `[batch_size,
/// glimpse_height, glimpse_width, channels]`.
class ExtractGlimpseV2 {
 public:
  /// Optional attribute setters for ExtractGlimpseV2
  struct Attrs {
    /// indicates if the offset coordinates are centered relative to
    /// the image, in which case the (0, 0) offset is relative to the center
    /// of the input images. If false, the (0,0) offset corresponds to the
    /// upper left corner of the input images.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs Centered(bool x) {
      Attrs ret = *this;
      ret.centered_ = x;
      return ret;
    }

    /// indicates if the offset coordinates are normalized.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs Normalized(bool x) {
      Attrs ret = *this;
      ret.normalized_ = x;
      return ret;
    }

    /// indicates if the noise should be generated using a
    /// uniform distribution or a Gaussian distribution.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UniformNoise(bool x) {
      Attrs ret = *this;
      ret.uniform_noise_ = x;
      return ret;
    }

    /// indicates if the noise should `uniform`, `gaussian`, or
    /// `zero`. The default is `uniform` which means the noise type
    /// will be decided by `uniform_noise`.
    ///
    /// Defaults to "uniform"
    TF_MUST_USE_RESULT Attrs Noise(StringPiece x) {
      Attrs ret = *this;
      ret.noise_ = x;
      return ret;
    }

    bool centered_ = true;
    bool normalized_ = true;
    bool uniform_noise_ = true;
    StringPiece noise_ = "uniform";
  };
  ExtractGlimpseV2(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                 ::tensorflow::Input size, ::tensorflow::Input offsets);
  ExtractGlimpseV2(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                 ::tensorflow::Input size, ::tensorflow::Input offsets, const
                 ExtractGlimpseV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return glimpse; }
  operator ::tensorflow::Input() const { return glimpse; }
  ::tensorflow::Node* node() const { return glimpse.node(); }

  static Attrs Centered(bool x) {
    return Attrs().Centered(x);
  }
  static Attrs Normalized(bool x) {
    return Attrs().Normalized(x);
  }
  static Attrs UniformNoise(bool x) {
    return Attrs().UniformNoise(x);
  }
  static Attrs Noise(StringPiece x) {
    return Attrs().Noise(x);
  }

  Operation operation;
  ::tensorflow::Output glimpse;
};

/// This op produces Region of Interests from given bounding boxes(bbox_deltas) encoded wrt anchors according to eq.2 in arXiv:1506.01497
///
///       The op selects top `pre_nms_topn` scoring boxes, decodes them with respect to anchors,
///       applies non-maximal suppression on overlapping boxes with higher than
///       `nms_threshold` intersection-over-union (iou) value, discarding boxes where shorter
///       side is less than `min_size`.
///       Inputs:
///       `scores`: A 4D tensor of shape [Batch, Height, Width, Num Anchors] containing the scores per anchor at given position
///       `bbox_deltas`: is a tensor of shape [Batch, Height, Width, 4 x Num Anchors] boxes encoded to each anchor
///       `anchors`: A 1D tensor of shape [4 x Num Anchors], representing the anchors.
///       Outputs:
///       `rois`: output RoIs, a 3D tensor of shape [Batch, post_nms_topn, 4], padded by 0 if less than post_nms_topn candidates found.
///       `roi_probabilities`: probability scores of each roi in 'rois', a 2D tensor of shape [Batch,post_nms_topn], padded with 0 if needed, sorted by scores.
///
/// Args:
/// * scope: A Scope object
/// * scores: A 4-D float tensor of shape `[num_images, height, width, num_achors]` containing scores of the boxes for given anchors, can be unsorted.
/// * bbox_deltas: A 4-D float tensor of shape `[num_images, height, width, 4 x num_anchors]`. encoding boxes with respec to each anchor.
/// Coordinates are given in the form [dy, dx, dh, dw].
/// * image_info: A 2-D float tensor of shape `[num_images, 5]` containing image information Height, Width, Scale.
/// * anchors: A 2-D float tensor of shape `[num_anchors, 4]` describing the anchor boxes. Boxes are formatted in the form [y1, x1, y2, x2].
/// * nms_threshold: A scalar float tensor for non-maximal-suppression threshold.
/// * pre_nms_topn: A scalar int tensor for the number of top scoring boxes to be used as input.
/// * min_size: A scalar float tensor. Any box that has a smaller size than min_size will be discarded.
///
/// Optional attributes (see `Attrs`):
/// * post_nms_topn: An integer. Maximum number of rois in the output.
///
/// Returns:
/// * `Output` rois: A 3-D float tensor of shape `[num_images,post_nms_topn,4]` representing the selected
/// region of interest boxes. Sorted in descending order in scores.
/// * `Output` roi_probabilities: A 2-D float tensor of shape `[num_images, post_nms_topn]` representing the score of the
/// region of interest box in `rois` tensor at the same index.
class GenerateBoundingBoxProposals {
 public:
  /// Optional attribute setters for GenerateBoundingBoxProposals
  struct Attrs {
    /// An integer. Maximum number of rois in the output.
    ///
    /// Defaults to 300
    TF_MUST_USE_RESULT Attrs PostNmsTopn(int64 x) {
      Attrs ret = *this;
      ret.post_nms_topn_ = x;
      return ret;
    }

    int64 post_nms_topn_ = 300;
  };
  GenerateBoundingBoxProposals(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input scores, ::tensorflow::Input
                             bbox_deltas, ::tensorflow::Input image_info,
                             ::tensorflow::Input anchors, ::tensorflow::Input
                             nms_threshold, ::tensorflow::Input pre_nms_topn,
                             ::tensorflow::Input min_size);
  GenerateBoundingBoxProposals(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input scores, ::tensorflow::Input
                             bbox_deltas, ::tensorflow::Input image_info,
                             ::tensorflow::Input anchors, ::tensorflow::Input
                             nms_threshold, ::tensorflow::Input pre_nms_topn,
                             ::tensorflow::Input min_size, const
                             GenerateBoundingBoxProposals::Attrs& attrs);

  static Attrs PostNmsTopn(int64 x) {
    return Attrs().PostNmsTopn(x);
  }

  Operation operation;
  ::tensorflow::Output rois;
  ::tensorflow::Output roi_probabilities;
};

/// Applies the given transform to each of the images.
///
/// If one row of `transforms` is `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps
/// the *output* point `(x, y)` to a transformed *input* point
/// `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
/// `k = c0 x + c1 y + 1`. If the transformed point lays outside of the input
/// image, the output pixel is set to 0.
///
/// Args:
/// * scope: A Scope object
/// * images: 4-D with shape `[batch, height, width, channels]`.
/// * transforms: 2-D Tensor, `[batch, 8]` or `[1, 8]` matrix, where each row corresponds to a 3 x 3
/// projective transformation matrix, with the last entry assumed to be 1. If there
/// is one row, the same transformation will be applied to all images.
/// * output_shape: 1-D Tensor [new_height, new_width].
/// * interpolation: Interpolation method, "NEAREST" or "BILINEAR".
///
/// Optional attributes (see `Attrs`):
/// * fill_mode: Fill mode, "REFLECT", "WRAP", or "CONSTANT".
///
/// Returns:
/// * `Output`: 4-D with shape
/// `[batch, new_height, new_width, channels]`.
class ImageProjectiveTransformV2 {
 public:
  /// Optional attribute setters for ImageProjectiveTransformV2
  struct Attrs {
    /// Fill mode, "REFLECT", "WRAP", or "CONSTANT".
    ///
    /// Defaults to "CONSTANT"
    TF_MUST_USE_RESULT Attrs FillMode(StringPiece x) {
      Attrs ret = *this;
      ret.fill_mode_ = x;
      return ret;
    }

    StringPiece fill_mode_ = "CONSTANT";
  };
  ImageProjectiveTransformV2(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input images, ::tensorflow::Input
                           transforms, ::tensorflow::Input output_shape,
                           StringPiece interpolation);
  ImageProjectiveTransformV2(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input images, ::tensorflow::Input
                           transforms, ::tensorflow::Input output_shape,
                           StringPiece interpolation, const
                           ImageProjectiveTransformV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return transformed_images; }
  operator ::tensorflow::Input() const { return transformed_images; }
  ::tensorflow::Node* node() const { return transformed_images.node(); }

  static Attrs FillMode(StringPiece x) {
    return Attrs().FillMode(x);
  }

  Operation operation;
  ::tensorflow::Output transformed_images;
};

/// Applies the given transform to each of the images.
///
/// If one row of `transforms` is `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps
/// the *output* point `(x, y)` to a transformed *input* point
/// `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
/// `k = c0 x + c1 y + 1`. If the transformed point lays outside of the input
/// image, the output pixel is set to fill_value.
///
/// Args:
/// * scope: A Scope object
/// * images: 4-D with shape `[batch, height, width, channels]`.
/// * transforms: 2-D Tensor, `[batch, 8]` or `[1, 8]` matrix, where each row corresponds to a 3 x 3
/// projective transformation matrix, with the last entry assumed to be 1. If there
/// is one row, the same transformation will be applied to all images.
/// * output_shape: 1-D Tensor [new_height, new_width].
/// * fill_value: float, the value to be filled when fill_mode is constant".
/// * interpolation: Interpolation method, "NEAREST" or "BILINEAR".
///
/// Optional attributes (see `Attrs`):
/// * fill_mode: Fill mode, "REFLECT", "WRAP", "CONSTANT", or "NEAREST".
///
/// Returns:
/// * `Output`: 4-D with shape
/// `[batch, new_height, new_width, channels]`.
class ImageProjectiveTransformV3 {
 public:
  /// Optional attribute setters for ImageProjectiveTransformV3
  struct Attrs {
    /// Fill mode, "REFLECT", "WRAP", "CONSTANT", or "NEAREST".
    ///
    /// Defaults to "CONSTANT"
    TF_MUST_USE_RESULT Attrs FillMode(StringPiece x) {
      Attrs ret = *this;
      ret.fill_mode_ = x;
      return ret;
    }

    StringPiece fill_mode_ = "CONSTANT";
  };
  ImageProjectiveTransformV3(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input images, ::tensorflow::Input
                           transforms, ::tensorflow::Input output_shape,
                           ::tensorflow::Input fill_value, StringPiece
                           interpolation);
  ImageProjectiveTransformV3(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input images, ::tensorflow::Input
                           transforms, ::tensorflow::Input output_shape,
                           ::tensorflow::Input fill_value, StringPiece
                           interpolation, const
                           ImageProjectiveTransformV3::Attrs& attrs);
  operator ::tensorflow::Output() const { return transformed_images; }
  operator ::tensorflow::Input() const { return transformed_images; }
  ::tensorflow::Node* node() const { return transformed_images.node(); }

  static Attrs FillMode(StringPiece x) {
    return Attrs().FillMode(x);
  }

  Operation operation;
  ::tensorflow::Output transformed_images;
};

/// Computes the gradient of bicubic interpolation.
///
/// Args:
/// * scope: A Scope object
/// * grads: 4-D with shape `[batch, height, width, channels]`.
/// * original_image: 4-D with shape `[batch, orig_height, orig_width, channels]`,
/// The image tensor that was resized.
///
/// Optional attributes (see `Attrs`):
/// * align_corners: If true, the centers of the 4 corner pixels of the input and grad tensors are
/// aligned. Defaults to false.
///
/// Returns:
/// * `Output`: 4-D with shape `[batch, orig_height, orig_width, channels]`.
/// Gradients with respect to the input image. Input image must have been
/// float or double.
class ResizeBicubicGrad {
 public:
  /// Optional attribute setters for ResizeBicubicGrad
  struct Attrs {
    /// If true, the centers of the 4 corner pixels of the input and grad tensors are
    /// aligned. Defaults to false.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs AlignCorners(bool x) {
      Attrs ret = *this;
      ret.align_corners_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs HalfPixelCenters(bool x) {
      Attrs ret = *this;
      ret.half_pixel_centers_ = x;
      return ret;
    }

    bool align_corners_ = false;
    bool half_pixel_centers_ = false;
  };
  ResizeBicubicGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input grads,
                  ::tensorflow::Input original_image);
  ResizeBicubicGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input grads,
                  ::tensorflow::Input original_image, const
                  ResizeBicubicGrad::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs AlignCorners(bool x) {
    return Attrs().AlignCorners(x);
  }
  static Attrs HalfPixelCenters(bool x) {
    return Attrs().HalfPixelCenters(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the gradient of bilinear interpolation.
///
/// Args:
/// * scope: A Scope object
/// * grads: 4-D with shape `[batch, height, width, channels]`.
/// * original_image: 4-D with shape `[batch, orig_height, orig_width, channels]`,
/// The image tensor that was resized.
///
/// Optional attributes (see `Attrs`):
/// * align_corners: If true, the centers of the 4 corner pixels of the input and grad tensors are
/// aligned. Defaults to false.
///
/// Returns:
/// * `Output`: 4-D with shape `[batch, orig_height, orig_width, channels]`.
/// Gradients with respect to the input image. Input image must have been
/// float or double.
class ResizeBilinearGrad {
 public:
  /// Optional attribute setters for ResizeBilinearGrad
  struct Attrs {
    /// If true, the centers of the 4 corner pixels of the input and grad tensors are
    /// aligned. Defaults to false.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs AlignCorners(bool x) {
      Attrs ret = *this;
      ret.align_corners_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs HalfPixelCenters(bool x) {
      Attrs ret = *this;
      ret.half_pixel_centers_ = x;
      return ret;
    }

    bool align_corners_ = false;
    bool half_pixel_centers_ = false;
  };
  ResizeBilinearGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input grads,
                   ::tensorflow::Input original_image);
  ResizeBilinearGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input grads,
                   ::tensorflow::Input original_image, const
                   ResizeBilinearGrad::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs AlignCorners(bool x) {
    return Attrs().AlignCorners(x);
  }
  static Attrs HalfPixelCenters(bool x) {
    return Attrs().HalfPixelCenters(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the gradient of nearest neighbor interpolation.
///
/// Args:
/// * scope: A Scope object
/// * grads: 4-D with shape `[batch, height, width, channels]`.
/// * size: = A 1-D int32 Tensor of 2 elements: `orig_height, orig_width`. The
/// original input size.
///
/// Optional attributes (see `Attrs`):
/// * align_corners: If true, the centers of the 4 corner pixels of the input and grad tensors are
/// aligned. Defaults to false.
///
/// Returns:
/// * `Output`: 4-D with shape `[batch, orig_height, orig_width, channels]`. Gradients
/// with respect to the input image.
class ResizeNearestNeighborGrad {
 public:
  /// Optional attribute setters for ResizeNearestNeighborGrad
  struct Attrs {
    /// If true, the centers of the 4 corner pixels of the input and grad tensors are
    /// aligned. Defaults to false.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs AlignCorners(bool x) {
      Attrs ret = *this;
      ret.align_corners_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs HalfPixelCenters(bool x) {
      Attrs ret = *this;
      ret.half_pixel_centers_ = x;
      return ret;
    }

    bool align_corners_ = false;
    bool half_pixel_centers_ = false;
  };
  ResizeNearestNeighborGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          grads, ::tensorflow::Input size);
  ResizeNearestNeighborGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          grads, ::tensorflow::Input size, const
                          ResizeNearestNeighborGrad::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs AlignCorners(bool x) {
    return Attrs().AlignCorners(x);
  }
  static Attrs HalfPixelCenters(bool x) {
    return Attrs().HalfPixelCenters(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class ScaleAndTranslateGrad {
 public:
  /// Optional attribute setters for ScaleAndTranslateGrad
  struct Attrs {
    /// Defaults to "lanczos3"
    TF_MUST_USE_RESULT Attrs KernelType(StringPiece x) {
      Attrs ret = *this;
      ret.kernel_type_ = x;
      return ret;
    }

    /// Defaults to true
    TF_MUST_USE_RESULT Attrs Antialias(bool x) {
      Attrs ret = *this;
      ret.antialias_ = x;
      return ret;
    }

    StringPiece kernel_type_ = "lanczos3";
    bool antialias_ = true;
  };
  ScaleAndTranslateGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      grads, ::tensorflow::Input original_image,
                      ::tensorflow::Input scale, ::tensorflow::Input
                      translation);
  ScaleAndTranslateGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      grads, ::tensorflow::Input original_image,
                      ::tensorflow::Input scale, ::tensorflow::Input
                      translation, const ScaleAndTranslateGrad::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs KernelType(StringPiece x) {
    return Attrs().KernelType(x);
  }
  static Attrs Antialias(bool x) {
    return Attrs().Antialias(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_IMAGE_OPS_INTERNAL_H_
