"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export

from typing import TypeVar, List, Any
from typing_extensions import Annotated

TV_AdjustContrast_T = TypeVar("TV_AdjustContrast_T", _atypes.Float32, _atypes.Float64, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt8)

def adjust_contrast(images: Annotated[Any, TV_AdjustContrast_T], contrast_factor: Annotated[Any, _atypes.Float32], min_value: Annotated[Any, _atypes.Float32], max_value: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Deprecated. Disallowed in GraphDef version >= 2.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`.
    contrast_factor: A `Tensor` of type `float32`.
    min_value: A `Tensor` of type `float32`.
    max_value: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AdjustContrast", name, images, contrast_factor, min_value,
        max_value)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return adjust_contrast_eager_fallback(
          images, contrast_factor, min_value, max_value, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AdjustContrast", images=images, contrast_factor=contrast_factor,
                          min_value=min_value, max_value=max_value, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AdjustContrast", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AdjustContrast = tf_export("raw_ops.AdjustContrast")(_ops.to_raw_op(adjust_contrast))


def adjust_contrast_eager_fallback(images: Annotated[Any, TV_AdjustContrast_T], contrast_factor: Annotated[Any, _atypes.Float32], min_value: Annotated[Any, _atypes.Float32], max_value: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Float32]:
  _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.uint8, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.float32, _dtypes.float64, ])
  contrast_factor = _ops.convert_to_tensor(contrast_factor, _dtypes.float32)
  min_value = _ops.convert_to_tensor(min_value, _dtypes.float32)
  max_value = _ops.convert_to_tensor(max_value, _dtypes.float32)
  _inputs_flat = [images, contrast_factor, min_value, max_value]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"AdjustContrast", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AdjustContrast", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_AdjustContrastv2_T = TypeVar("TV_AdjustContrastv2_T", _atypes.Float32, _atypes.Half)

def adjust_contrastv2(images: Annotated[Any, TV_AdjustContrastv2_T], contrast_factor: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, TV_AdjustContrastv2_T]:
  r"""Adjust the contrast of one or more images.

  `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
  interpreted as `[height, width, channels]`.  The other dimensions only
  represent a collection of images, such as `[batch, height, width, channels].`

  Contrast is adjusted independently for each channel of each image.

  For each channel, the Op first computes the mean of the image pixels in the
  channel and then adjusts each component of each pixel to
  `(x - mean) * contrast_factor + mean`.

  Args:
    images: A `Tensor`. Must be one of the following types: `half`, `float32`.
      Images to adjust.  At least 3-D.
    contrast_factor: A `Tensor` of type `float32`.
      A float multiplier for adjusting contrast.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AdjustContrastv2", name, images, contrast_factor)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return adjust_contrastv2_eager_fallback(
          images, contrast_factor, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AdjustContrastv2", images=images, contrast_factor=contrast_factor,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AdjustContrastv2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AdjustContrastv2 = tf_export("raw_ops.AdjustContrastv2")(_ops.to_raw_op(adjust_contrastv2))


def adjust_contrastv2_eager_fallback(images: Annotated[Any, TV_AdjustContrastv2_T], contrast_factor: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, TV_AdjustContrastv2_T]:
  _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.half, _dtypes.float32, ], _dtypes.float32)
  contrast_factor = _ops.convert_to_tensor(contrast_factor, _dtypes.float32)
  _inputs_flat = [images, contrast_factor]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"AdjustContrastv2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AdjustContrastv2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_AdjustHue_T = TypeVar("TV_AdjustHue_T", _atypes.Float32, _atypes.Half)

def adjust_hue(images: Annotated[Any, TV_AdjustHue_T], delta: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, TV_AdjustHue_T]:
  r"""Adjust the hue of one or more images.

  `images` is a tensor of at least 3 dimensions.  The last dimension is
  interpreted as channels, and must be three.

  The input image is considered in the RGB colorspace. Conceptually, the RGB
  colors are first mapped into HSV. A delta is then applied all the hue values,
  and then remapped back to RGB colorspace.

  Args:
    images: A `Tensor`. Must be one of the following types: `half`, `float32`.
      Images to adjust.  At least 3-D.
    delta: A `Tensor` of type `float32`. A float delta to add to the hue.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AdjustHue", name, images, delta)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return adjust_hue_eager_fallback(
          images, delta, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AdjustHue", images=images, delta=delta, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AdjustHue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AdjustHue = tf_export("raw_ops.AdjustHue")(_ops.to_raw_op(adjust_hue))


def adjust_hue_eager_fallback(images: Annotated[Any, TV_AdjustHue_T], delta: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, TV_AdjustHue_T]:
  _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.half, _dtypes.float32, ], _dtypes.float32)
  delta = _ops.convert_to_tensor(delta, _dtypes.float32)
  _inputs_flat = [images, delta]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"AdjustHue", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AdjustHue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_AdjustSaturation_T = TypeVar("TV_AdjustSaturation_T", _atypes.Float32, _atypes.Half)

def adjust_saturation(images: Annotated[Any, TV_AdjustSaturation_T], scale: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, TV_AdjustSaturation_T]:
  r"""Adjust the saturation of one or more images.

  `images` is a tensor of at least 3 dimensions.  The last dimension is
  interpreted as channels, and must be three.

  The input image is considered in the RGB colorspace. Conceptually, the RGB
  colors are first mapped into HSV. A scale is then applied all the saturation
  values, and then remapped back to RGB colorspace.

  Args:
    images: A `Tensor`. Must be one of the following types: `half`, `float32`.
      Images to adjust.  At least 3-D.
    scale: A `Tensor` of type `float32`.
      A float scale to add to the saturation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AdjustSaturation", name, images, scale)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return adjust_saturation_eager_fallback(
          images, scale, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AdjustSaturation", images=images, scale=scale, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AdjustSaturation", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AdjustSaturation = tf_export("raw_ops.AdjustSaturation")(_ops.to_raw_op(adjust_saturation))


def adjust_saturation_eager_fallback(images: Annotated[Any, TV_AdjustSaturation_T], scale: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, TV_AdjustSaturation_T]:
  _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.half, _dtypes.float32, ], _dtypes.float32)
  scale = _ops.convert_to_tensor(scale, _dtypes.float32)
  _inputs_flat = [images, scale]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"AdjustSaturation", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AdjustSaturation", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_CombinedNonMaxSuppressionOutput = collections.namedtuple(
    "CombinedNonMaxSuppression",
    ["nmsed_boxes", "nmsed_scores", "nmsed_classes", "valid_detections"])


def combined_non_max_suppression(boxes: Annotated[Any, _atypes.Float32], scores: Annotated[Any, _atypes.Float32], max_output_size_per_class: Annotated[Any, _atypes.Int32], max_total_size: Annotated[Any, _atypes.Int32], iou_threshold: Annotated[Any, _atypes.Float32], score_threshold: Annotated[Any, _atypes.Float32], pad_per_class:bool=False, clip_boxes:bool=True, name=None):
  r"""Greedily selects a subset of bounding boxes in descending order of score,

  This operation performs non_max_suppression on the inputs per batch, across
  all classes.
  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes are supplied as
  [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system. Also note that
  this algorithm is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is the final boxes, scores and classes tensor
  returned after performing non_max_suppression.

  Args:
    boxes: A `Tensor` of type `float32`.
      A 4-D float tensor of shape `[batch_size, num_boxes, q, 4]`. If `q` is 1 then
      same boxes are used for all classes otherwise, if `q` is equal to number of
      classes, class-specific boxes are used.
    scores: A `Tensor` of type `float32`.
      A 3-D float tensor of shape `[batch_size, num_boxes, num_classes]`
      representing a single score corresponding to each box (each row of boxes).
    max_output_size_per_class: A `Tensor` of type `int32`.
      A scalar integer tensor representing the maximum number of
      boxes to be selected by non max suppression per class
    max_total_size: A `Tensor` of type `int32`.
      An int32 scalar representing the maximum number of boxes retained over all
      classes. Note that setting this value to a large number may result in OOM error
      depending on the system workload.
    iou_threshold: A `Tensor` of type `float32`.
      A 0-D float tensor representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    score_threshold: A `Tensor` of type `float32`.
      A 0-D float tensor representing the threshold for deciding when to remove
      boxes based on score.
    pad_per_class: An optional `bool`. Defaults to `False`.
      If false, the output nmsed boxes, scores and classes
      are padded/clipped to `max_total_size`. If true, the
      output nmsed boxes, scores and classes are padded to be of length
      `max_size_per_class`*`num_classes`, unless it exceeds `max_total_size` in
      which case it is clipped to `max_total_size`. Defaults to false.
    clip_boxes: An optional `bool`. Defaults to `True`.
      If true, assume the box coordinates are between [0, 1] and clip the output boxes
      if they fall beyond [0, 1]. If false, do not do clipping and output the box
      coordinates as it is.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections).

    nmsed_boxes: A `Tensor` of type `float32`.
    nmsed_scores: A `Tensor` of type `float32`.
    nmsed_classes: A `Tensor` of type `float32`.
    valid_detections: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CombinedNonMaxSuppression", name, boxes, scores,
        max_output_size_per_class, max_total_size, iou_threshold,
        score_threshold, "pad_per_class", pad_per_class, "clip_boxes",
        clip_boxes)
      _result = _CombinedNonMaxSuppressionOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return combined_non_max_suppression_eager_fallback(
          boxes, scores, max_output_size_per_class, max_total_size,
          iou_threshold, score_threshold, pad_per_class=pad_per_class,
          clip_boxes=clip_boxes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if pad_per_class is None:
    pad_per_class = False
  pad_per_class = _execute.make_bool(pad_per_class, "pad_per_class")
  if clip_boxes is None:
    clip_boxes = True
  clip_boxes = _execute.make_bool(clip_boxes, "clip_boxes")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CombinedNonMaxSuppression", boxes=boxes, scores=scores,
                                     max_output_size_per_class=max_output_size_per_class,
                                     max_total_size=max_total_size,
                                     iou_threshold=iou_threshold,
                                     score_threshold=score_threshold,
                                     pad_per_class=pad_per_class,
                                     clip_boxes=clip_boxes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("pad_per_class", _op._get_attr_bool("pad_per_class"),
              "clip_boxes", _op._get_attr_bool("clip_boxes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CombinedNonMaxSuppression", _inputs_flat, _attrs, _result)
  _result = _CombinedNonMaxSuppressionOutput._make(_result)
  return _result

CombinedNonMaxSuppression = tf_export("raw_ops.CombinedNonMaxSuppression")(_ops.to_raw_op(combined_non_max_suppression))


def combined_non_max_suppression_eager_fallback(boxes: Annotated[Any, _atypes.Float32], scores: Annotated[Any, _atypes.Float32], max_output_size_per_class: Annotated[Any, _atypes.Int32], max_total_size: Annotated[Any, _atypes.Int32], iou_threshold: Annotated[Any, _atypes.Float32], score_threshold: Annotated[Any, _atypes.Float32], pad_per_class: bool, clip_boxes: bool, name, ctx):
  if pad_per_class is None:
    pad_per_class = False
  pad_per_class = _execute.make_bool(pad_per_class, "pad_per_class")
  if clip_boxes is None:
    clip_boxes = True
  clip_boxes = _execute.make_bool(clip_boxes, "clip_boxes")
  boxes = _ops.convert_to_tensor(boxes, _dtypes.float32)
  scores = _ops.convert_to_tensor(scores, _dtypes.float32)
  max_output_size_per_class = _ops.convert_to_tensor(max_output_size_per_class, _dtypes.int32)
  max_total_size = _ops.convert_to_tensor(max_total_size, _dtypes.int32)
  iou_threshold = _ops.convert_to_tensor(iou_threshold, _dtypes.float32)
  score_threshold = _ops.convert_to_tensor(score_threshold, _dtypes.float32)
  _inputs_flat = [boxes, scores, max_output_size_per_class, max_total_size, iou_threshold, score_threshold]
  _attrs = ("pad_per_class", pad_per_class, "clip_boxes", clip_boxes)
  _result = _execute.execute(b"CombinedNonMaxSuppression", 4,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CombinedNonMaxSuppression", _inputs_flat, _attrs, _result)
  _result = _CombinedNonMaxSuppressionOutput._make(_result)
  return _result


TV_CropAndResize_T = TypeVar("TV_CropAndResize_T", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt8)

def crop_and_resize(image: Annotated[Any, TV_CropAndResize_T], boxes: Annotated[Any, _atypes.Float32], box_ind: Annotated[Any, _atypes.Int32], crop_size: Annotated[Any, _atypes.Int32], method:str="bilinear", extrapolation_value:float=0, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Extracts crops from the input image tensor and resizes them.

  Extracts crops from the input image tensor and resizes them using bilinear
  sampling or nearest neighbor sampling (possibly with aspect ratio change) to a
  common output size specified by `crop_size`. This is more general than the
  `crop_to_bounding_box` op which extracts a fixed size slice from the input image
  and does not allow resizing or aspect ratio change.

  Returns a tensor with `crops` from the input `image` at positions defined at the
  bounding box locations in `boxes`. The cropped boxes are all resized (with
  bilinear or nearest neighbor interpolation) to a fixed
  `size = [crop_height, crop_width]`. The result is a 4-D tensor
  `[num_boxes, crop_height, crop_width, depth]`. The resizing is corner aligned.
  In particular, if `boxes = [[0, 0, 1, 1]]`, the method will give identical
  results to using `tf.image.resize_bilinear()` or
  `tf.image.resize_nearest_neighbor()`(depends on the `method` argument) with
  `align_corners=True`.

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `uint16`, `int8`, `int16`, `int32`, `int64`, `half`, `float32`, `float64`.
      A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A `Tensor` of type `float32`.
      A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
      specifies the coordinates of a box in the `box_ind[i]` image and is specified
      in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
      `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
      `[0, 1]` interval of normalized image height is mapped to
      `[0, image_height - 1]` in image height coordinates. We do allow `y1` > `y2`, in
      which case the sampled crop is an up-down flipped version of the original
      image. The width dimension is treated similarly. Normalized coordinates
      outside the `[0, 1]` range are allowed, in which case we use
      `extrapolation_value` to extrapolate the input image values.
    box_ind: A `Tensor` of type `int32`.
      A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
      The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
    crop_size: A `Tensor` of type `int32`.
      A 1-D tensor of 2 elements, `size = [crop_height, crop_width]`. All
      cropped image patches are resized to this size. The aspect ratio of the image
      content is not preserved. Both `crop_height` and `crop_width` need to be
      positive.
    method: An optional `string` from: `"bilinear", "nearest"`. Defaults to `"bilinear"`.
      A string specifying the sampling method for resizing. It can be either
      `"bilinear"` or `"nearest"` and default to `"bilinear"`. Currently two sampling
      methods are supported: Bilinear and Nearest Neighbor.
    extrapolation_value: An optional `float`. Defaults to `0`.
      Value used for extrapolation, when applicable.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CropAndResize", name, image, boxes, box_ind, crop_size,
        "method", method, "extrapolation_value", extrapolation_value)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return crop_and_resize_eager_fallback(
          image, boxes, box_ind, crop_size, method=method,
          extrapolation_value=extrapolation_value, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if method is None:
    method = "bilinear"
  method = _execute.make_str(method, "method")
  if extrapolation_value is None:
    extrapolation_value = 0
  extrapolation_value = _execute.make_float(extrapolation_value, "extrapolation_value")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CropAndResize", image=image, boxes=boxes, box_ind=box_ind,
                         crop_size=crop_size, method=method,
                         extrapolation_value=extrapolation_value, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "method", _op.get_attr("method"),
              "extrapolation_value", _op.get_attr("extrapolation_value"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CropAndResize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CropAndResize = tf_export("raw_ops.CropAndResize")(_ops.to_raw_op(crop_and_resize))


def crop_and_resize_eager_fallback(image: Annotated[Any, TV_CropAndResize_T], boxes: Annotated[Any, _atypes.Float32], box_ind: Annotated[Any, _atypes.Int32], crop_size: Annotated[Any, _atypes.Int32], method: str, extrapolation_value: float, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if method is None:
    method = "bilinear"
  method = _execute.make_str(method, "method")
  if extrapolation_value is None:
    extrapolation_value = 0
  extrapolation_value = _execute.make_float(extrapolation_value, "extrapolation_value")
  _attr_T, (image,) = _execute.args_to_matching_eager([image], ctx, [_dtypes.uint8, _dtypes.uint16, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  boxes = _ops.convert_to_tensor(boxes, _dtypes.float32)
  box_ind = _ops.convert_to_tensor(box_ind, _dtypes.int32)
  crop_size = _ops.convert_to_tensor(crop_size, _dtypes.int32)
  _inputs_flat = [image, boxes, box_ind, crop_size]
  _attrs = ("T", _attr_T, "method", method, "extrapolation_value",
  extrapolation_value)
  _result = _execute.execute(b"CropAndResize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CropAndResize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CropAndResizeGradBoxes_T = TypeVar("TV_CropAndResizeGradBoxes_T", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt8)

def crop_and_resize_grad_boxes(grads: Annotated[Any, _atypes.Float32], image: Annotated[Any, TV_CropAndResizeGradBoxes_T], boxes: Annotated[Any, _atypes.Float32], box_ind: Annotated[Any, _atypes.Int32], method:str="bilinear", name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Computes the gradient of the crop_and_resize op wrt the input boxes tensor.

  Args:
    grads: A `Tensor` of type `float32`.
      A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
    image: A `Tensor`. Must be one of the following types: `uint8`, `uint16`, `int8`, `int16`, `int32`, `int64`, `half`, `float32`, `float64`.
      A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A `Tensor` of type `float32`.
      A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
      specifies the coordinates of a box in the `box_ind[i]` image and is specified
      in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
      `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
      `[0, 1]` interval of normalized image height is mapped to
      `[0, image_height - 1] in image height coordinates. We do allow y1 > y2, in
      which case the sampled crop is an up-down flipped version of the original
      image. The width dimension is treated similarly. Normalized coordinates
      outside the `[0, 1]` range are allowed, in which case we use
      `extrapolation_value` to extrapolate the input image values.
    box_ind: A `Tensor` of type `int32`.
      A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
      The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
    method: An optional `string` from: `"bilinear"`. Defaults to `"bilinear"`.
      A string specifying the interpolation method. Only 'bilinear' is
      supported for now.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CropAndResizeGradBoxes", name, grads, image, boxes, box_ind,
        "method", method)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return crop_and_resize_grad_boxes_eager_fallback(
          grads, image, boxes, box_ind, method=method, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if method is None:
    method = "bilinear"
  method = _execute.make_str(method, "method")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CropAndResizeGradBoxes", grads=grads, image=image, boxes=boxes,
                                  box_ind=box_ind, method=method, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "method", _op.get_attr("method"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CropAndResizeGradBoxes", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CropAndResizeGradBoxes = tf_export("raw_ops.CropAndResizeGradBoxes")(_ops.to_raw_op(crop_and_resize_grad_boxes))


def crop_and_resize_grad_boxes_eager_fallback(grads: Annotated[Any, _atypes.Float32], image: Annotated[Any, TV_CropAndResizeGradBoxes_T], boxes: Annotated[Any, _atypes.Float32], box_ind: Annotated[Any, _atypes.Int32], method: str, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if method is None:
    method = "bilinear"
  method = _execute.make_str(method, "method")
  _attr_T, (image,) = _execute.args_to_matching_eager([image], ctx, [_dtypes.uint8, _dtypes.uint16, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  grads = _ops.convert_to_tensor(grads, _dtypes.float32)
  boxes = _ops.convert_to_tensor(boxes, _dtypes.float32)
  box_ind = _ops.convert_to_tensor(box_ind, _dtypes.int32)
  _inputs_flat = [grads, image, boxes, box_ind]
  _attrs = ("T", _attr_T, "method", method)
  _result = _execute.execute(b"CropAndResizeGradBoxes", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CropAndResizeGradBoxes", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CropAndResizeGradImage_T = TypeVar("TV_CropAndResizeGradImage_T", _atypes.Float32, _atypes.Float64, _atypes.Half)

def crop_and_resize_grad_image(grads: Annotated[Any, _atypes.Float32], boxes: Annotated[Any, _atypes.Float32], box_ind: Annotated[Any, _atypes.Int32], image_size: Annotated[Any, _atypes.Int32], T: TV_CropAndResizeGradImage_T, method:str="bilinear", name=None) -> Annotated[Any, TV_CropAndResizeGradImage_T]:
  r"""Computes the gradient of the crop_and_resize op wrt the input image tensor.

  Args:
    grads: A `Tensor` of type `float32`.
      A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
    boxes: A `Tensor` of type `float32`.
      A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
      specifies the coordinates of a box in the `box_ind[i]` image and is specified
      in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
      `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
      `[0, 1]` interval of normalized image height is mapped to
      `[0, image_height - 1] in image height coordinates. We do allow y1 > y2, in
      which case the sampled crop is an up-down flipped version of the original
      image. The width dimension is treated similarly. Normalized coordinates
      outside the `[0, 1]` range are allowed, in which case we use
      `extrapolation_value` to extrapolate the input image values.
    box_ind: A `Tensor` of type `int32`.
      A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
      The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
    image_size: A `Tensor` of type `int32`.
      A 1-D tensor with value `[batch, image_height, image_width, depth]`
      containing the original image size. Both `image_height` and `image_width` need
      to be positive.
    T: A `tf.DType` from: `tf.float32, tf.half, tf.float64`.
    method: An optional `string` from: `"bilinear", "nearest"`. Defaults to `"bilinear"`.
      A string specifying the interpolation method. Only 'bilinear' is
      supported for now.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `T`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CropAndResizeGradImage", name, grads, boxes, box_ind,
        image_size, "T", T, "method", method)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return crop_and_resize_grad_image_eager_fallback(
          grads, boxes, box_ind, image_size, T=T, method=method, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  T = _execute.make_type(T, "T")
  if method is None:
    method = "bilinear"
  method = _execute.make_str(method, "method")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CropAndResizeGradImage", grads=grads, boxes=boxes, box_ind=box_ind,
                                  image_size=image_size, T=T, method=method,
                                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "method", _op.get_attr("method"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CropAndResizeGradImage", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CropAndResizeGradImage = tf_export("raw_ops.CropAndResizeGradImage")(_ops.to_raw_op(crop_and_resize_grad_image))


def crop_and_resize_grad_image_eager_fallback(grads: Annotated[Any, _atypes.Float32], boxes: Annotated[Any, _atypes.Float32], box_ind: Annotated[Any, _atypes.Int32], image_size: Annotated[Any, _atypes.Int32], T: TV_CropAndResizeGradImage_T, method: str, name, ctx) -> Annotated[Any, TV_CropAndResizeGradImage_T]:
  T = _execute.make_type(T, "T")
  if method is None:
    method = "bilinear"
  method = _execute.make_str(method, "method")
  grads = _ops.convert_to_tensor(grads, _dtypes.float32)
  boxes = _ops.convert_to_tensor(boxes, _dtypes.float32)
  box_ind = _ops.convert_to_tensor(box_ind, _dtypes.int32)
  image_size = _ops.convert_to_tensor(image_size, _dtypes.int32)
  _inputs_flat = [grads, boxes, box_ind, image_size]
  _attrs = ("T", T, "method", method)
  _result = _execute.execute(b"CropAndResizeGradImage", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CropAndResizeGradImage", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def decode_and_crop_jpeg(contents: Annotated[Any, _atypes.String], crop_window: Annotated[Any, _atypes.Int32], channels:int=0, ratio:int=1, fancy_upscaling:bool=True, try_recover_truncated:bool=False, acceptable_fraction:float=1, dct_method:str="", name=None) -> Annotated[Any, _atypes.UInt8]:
  r"""Decode and Crop a JPEG-encoded image to a uint8 tensor.

  The attr `channels` indicates the desired number of color channels for the
  decoded image.

  Accepted values are:

  *   0: Use the number of channels in the JPEG-encoded image.
  *   1: output a grayscale image.
  *   3: output an RGB image.

  If needed, the JPEG-encoded image is transformed to match the requested number
  of color channels.

  The attr `ratio` allows downscaling the image by an integer factor during
  decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
  downscaling the image later.


  It is equivalent to a combination of decode and crop, but much faster by only
  decoding partial jpeg image.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The JPEG-encoded image.
    crop_window: A `Tensor` of type `int32`.
      1-D.  The crop window: [crop_y, crop_x, crop_height, crop_width].
    channels: An optional `int`. Defaults to `0`.
      Number of color channels for the decoded image.
    ratio: An optional `int`. Defaults to `1`. Downscaling ratio.
    fancy_upscaling: An optional `bool`. Defaults to `True`.
      If true use a slower but nicer upscaling of the
      chroma planes (yuv420/422 only).
    try_recover_truncated: An optional `bool`. Defaults to `False`.
      If true try to recover an image from truncated input.
    acceptable_fraction: An optional `float`. Defaults to `1`.
      The minimum required fraction of lines before a truncated
      input is accepted.
    dct_method: An optional `string`. Defaults to `""`.
      string specifying a hint about the algorithm used for
      decompression.  Defaults to "" which maps to a system-specific
      default.  Currently valid values are ["INTEGER_FAST",
      "INTEGER_ACCURATE"].  The hint may be ignored (e.g., the internal
      jpeg library changes to a version that does not have that specific
      option.)
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DecodeAndCropJpeg", name, contents, crop_window, "channels",
        channels, "ratio", ratio, "fancy_upscaling", fancy_upscaling,
        "try_recover_truncated", try_recover_truncated, "acceptable_fraction",
        acceptable_fraction, "dct_method", dct_method)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return decode_and_crop_jpeg_eager_fallback(
          contents, crop_window, channels=channels, ratio=ratio,
          fancy_upscaling=fancy_upscaling,
          try_recover_truncated=try_recover_truncated,
          acceptable_fraction=acceptable_fraction, dct_method=dct_method,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if channels is None:
    channels = 0
  channels = _execute.make_int(channels, "channels")
  if ratio is None:
    ratio = 1
  ratio = _execute.make_int(ratio, "ratio")
  if fancy_upscaling is None:
    fancy_upscaling = True
  fancy_upscaling = _execute.make_bool(fancy_upscaling, "fancy_upscaling")
  if try_recover_truncated is None:
    try_recover_truncated = False
  try_recover_truncated = _execute.make_bool(try_recover_truncated, "try_recover_truncated")
  if acceptable_fraction is None:
    acceptable_fraction = 1
  acceptable_fraction = _execute.make_float(acceptable_fraction, "acceptable_fraction")
  if dct_method is None:
    dct_method = ""
  dct_method = _execute.make_str(dct_method, "dct_method")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DecodeAndCropJpeg", contents=contents, crop_window=crop_window,
                             channels=channels, ratio=ratio,
                             fancy_upscaling=fancy_upscaling,
                             try_recover_truncated=try_recover_truncated,
                             acceptable_fraction=acceptable_fraction,
                             dct_method=dct_method, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("channels", _op._get_attr_int("channels"), "ratio",
              _op._get_attr_int("ratio"), "fancy_upscaling",
              _op._get_attr_bool("fancy_upscaling"), "try_recover_truncated",
              _op._get_attr_bool("try_recover_truncated"),
              "acceptable_fraction", _op.get_attr("acceptable_fraction"),
              "dct_method", _op.get_attr("dct_method"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DecodeAndCropJpeg", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DecodeAndCropJpeg = tf_export("raw_ops.DecodeAndCropJpeg")(_ops.to_raw_op(decode_and_crop_jpeg))


def decode_and_crop_jpeg_eager_fallback(contents: Annotated[Any, _atypes.String], crop_window: Annotated[Any, _atypes.Int32], channels: int, ratio: int, fancy_upscaling: bool, try_recover_truncated: bool, acceptable_fraction: float, dct_method: str, name, ctx) -> Annotated[Any, _atypes.UInt8]:
  if channels is None:
    channels = 0
  channels = _execute.make_int(channels, "channels")
  if ratio is None:
    ratio = 1
  ratio = _execute.make_int(ratio, "ratio")
  if fancy_upscaling is None:
    fancy_upscaling = True
  fancy_upscaling = _execute.make_bool(fancy_upscaling, "fancy_upscaling")
  if try_recover_truncated is None:
    try_recover_truncated = False
  try_recover_truncated = _execute.make_bool(try_recover_truncated, "try_recover_truncated")
  if acceptable_fraction is None:
    acceptable_fraction = 1
  acceptable_fraction = _execute.make_float(acceptable_fraction, "acceptable_fraction")
  if dct_method is None:
    dct_method = ""
  dct_method = _execute.make_str(dct_method, "dct_method")
  contents = _ops.convert_to_tensor(contents, _dtypes.string)
  crop_window = _ops.convert_to_tensor(crop_window, _dtypes.int32)
  _inputs_flat = [contents, crop_window]
  _attrs = ("channels", channels, "ratio", ratio, "fancy_upscaling",
  fancy_upscaling, "try_recover_truncated", try_recover_truncated,
  "acceptable_fraction", acceptable_fraction, "dct_method", dct_method)
  _result = _execute.execute(b"DecodeAndCropJpeg", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DecodeAndCropJpeg", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def decode_bmp(contents: Annotated[Any, _atypes.String], channels:int=0, name=None) -> Annotated[Any, _atypes.UInt8]:
  r"""Decode the first frame of a BMP-encoded image to a uint8 tensor.

  The attr `channels` indicates the desired number of color channels for the
  decoded image.

  Accepted values are:

  *   0: Use the number of channels in the BMP-encoded image.
  *   3: output an RGB image.
  *   4: output an RGBA image.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The BMP-encoded image.
    channels: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DecodeBmp", name, contents, "channels", channels)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return decode_bmp_eager_fallback(
          contents, channels=channels, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if channels is None:
    channels = 0
  channels = _execute.make_int(channels, "channels")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DecodeBmp", contents=contents, channels=channels, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("channels", _op._get_attr_int("channels"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DecodeBmp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DecodeBmp = tf_export("raw_ops.DecodeBmp")(_ops.to_raw_op(decode_bmp))


def decode_bmp_eager_fallback(contents: Annotated[Any, _atypes.String], channels: int, name, ctx) -> Annotated[Any, _atypes.UInt8]:
  if channels is None:
    channels = 0
  channels = _execute.make_int(channels, "channels")
  contents = _ops.convert_to_tensor(contents, _dtypes.string)
  _inputs_flat = [contents]
  _attrs = ("channels", channels)
  _result = _execute.execute(b"DecodeBmp", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DecodeBmp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def decode_gif(contents: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.UInt8]:
  r"""Decode the frame(s) of a GIF-encoded image to a uint8 tensor.

  GIF images with frame or transparency compression are not supported.
  On Linux and MacOS systems, convert animated GIFs from compressed to
  uncompressed by running:

      convert $src.gif -coalesce $dst.gif

  This op also supports decoding JPEGs and PNGs, though it is cleaner to use
  `tf.io.decode_image`.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The GIF-encoded image.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DecodeGif", name, contents)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return decode_gif_eager_fallback(
          contents, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DecodeGif", contents=contents, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DecodeGif", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DecodeGif = tf_export("raw_ops.DecodeGif")(_ops.to_raw_op(decode_gif))


def decode_gif_eager_fallback(contents: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.UInt8]:
  contents = _ops.convert_to_tensor(contents, _dtypes.string)
  _inputs_flat = [contents]
  _attrs = None
  _result = _execute.execute(b"DecodeGif", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DecodeGif", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DecodeImage_dtype = TypeVar("TV_DecodeImage_dtype", _atypes.Float32, _atypes.UInt16, _atypes.UInt8)

def decode_image(contents: Annotated[Any, _atypes.String], channels:int=0, dtype:TV_DecodeImage_dtype=_dtypes.uint8, expand_animations:bool=True, name=None) -> Annotated[Any, TV_DecodeImage_dtype]:
  r"""Function for decode_bmp, decode_gif, decode_jpeg, and decode_png.

  Detects whether an image is a BMP, GIF, JPEG, or PNG, and performs the
  appropriate operation to convert the input bytes string into a Tensor of type
  dtype.

  *NOTE*: decode_gif returns a 4-D array [num_frames, height, width, 3], as
  opposed to decode_bmp, decode_jpeg and decode_png, which return 3-D arrays
  [height, width, num_channels]. Make sure to take this into account when
  constructing your graph if you are intermixing GIF files with BMP, JPEG, and/or
  PNG files. Alternately, set the expand_animations argument of this function to
  False, in which case the op will return 3-dimensional tensors and will truncate
  animated GIF files to the first frame.

  *NOTE*: If the first frame of an animated GIF does not occupy the entire
  canvas (maximum frame width x maximum frame height), then it fills the
  unoccupied areas (in the first frame) with zeros (black). For frames after the
  first frame that does not occupy the entire canvas, it uses the previous
  frame to fill the unoccupied areas.

  Args:
    contents: A `Tensor` of type `string`. 0-D. The encoded image bytes.
    channels: An optional `int`. Defaults to `0`.
      Number of color channels for the decoded image.
    dtype: An optional `tf.DType` from: `tf.uint8, tf.uint16, tf.float32`. Defaults to `tf.uint8`.
      The desired DType of the returned Tensor.
    expand_animations: An optional `bool`. Defaults to `True`.
      Controls the output shape of the returned op. If True, the returned op will
      produce a 3-D tensor for PNG, JPEG, and BMP files; and a 4-D tensor for all
      GIFs, whether animated or not. If, False, the returned op will produce a 3-D
      tensor for all file types and will truncate animated GIFs to the first frame.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DecodeImage", name, contents, "channels", channels, "dtype",
        dtype, "expand_animations", expand_animations)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return decode_image_eager_fallback(
          contents, channels=channels, dtype=dtype,
          expand_animations=expand_animations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if channels is None:
    channels = 0
  channels = _execute.make_int(channels, "channels")
  if dtype is None:
    dtype = _dtypes.uint8
  dtype = _execute.make_type(dtype, "dtype")
  if expand_animations is None:
    expand_animations = True
  expand_animations = _execute.make_bool(expand_animations, "expand_animations")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DecodeImage", contents=contents, channels=channels, dtype=dtype,
                       expand_animations=expand_animations, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("channels", _op._get_attr_int("channels"), "dtype",
              _op._get_attr_type("dtype"), "expand_animations",
              _op._get_attr_bool("expand_animations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DecodeImage", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DecodeImage = tf_export("raw_ops.DecodeImage")(_ops.to_raw_op(decode_image))


def decode_image_eager_fallback(contents: Annotated[Any, _atypes.String], channels: int, dtype: TV_DecodeImage_dtype, expand_animations: bool, name, ctx) -> Annotated[Any, TV_DecodeImage_dtype]:
  if channels is None:
    channels = 0
  channels = _execute.make_int(channels, "channels")
  if dtype is None:
    dtype = _dtypes.uint8
  dtype = _execute.make_type(dtype, "dtype")
  if expand_animations is None:
    expand_animations = True
  expand_animations = _execute.make_bool(expand_animations, "expand_animations")
  contents = _ops.convert_to_tensor(contents, _dtypes.string)
  _inputs_flat = [contents]
  _attrs = ("channels", channels, "dtype", dtype, "expand_animations",
  expand_animations)
  _result = _execute.execute(b"DecodeImage", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DecodeImage", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def decode_jpeg(contents: Annotated[Any, _atypes.String], channels:int=0, ratio:int=1, fancy_upscaling:bool=True, try_recover_truncated:bool=False, acceptable_fraction:float=1, dct_method:str="", name=None) -> Annotated[Any, _atypes.UInt8]:
  r"""Decode a JPEG-encoded image to a uint8 tensor.

  The attr `channels` indicates the desired number of color channels for the
  decoded image.

  Accepted values are:

  *   0: Use the number of channels in the JPEG-encoded image.
  *   1: output a grayscale image.
  *   3: output an RGB image.

  If needed, the JPEG-encoded image is transformed to match the requested number
  of color channels.

  The attr `ratio` allows downscaling the image by an integer factor during
  decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
  downscaling the image later.


  This op also supports decoding PNGs and non-animated GIFs since the interface is
  the same, though it is cleaner to use `tf.io.decode_image`.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The JPEG-encoded image.
    channels: An optional `int`. Defaults to `0`.
      Number of color channels for the decoded image.
    ratio: An optional `int`. Defaults to `1`. Downscaling ratio.
    fancy_upscaling: An optional `bool`. Defaults to `True`.
      If true use a slower but nicer upscaling of the
      chroma planes (yuv420/422 only).
    try_recover_truncated: An optional `bool`. Defaults to `False`.
      If true try to recover an image from truncated input.
    acceptable_fraction: An optional `float`. Defaults to `1`.
      The minimum required fraction of lines before a truncated
      input is accepted.
    dct_method: An optional `string`. Defaults to `""`.
      string specifying a hint about the algorithm used for
      decompression.  Defaults to "" which maps to a system-specific
      default.  Currently valid values are ["INTEGER_FAST",
      "INTEGER_ACCURATE"].  The hint may be ignored (e.g., the internal
      jpeg library changes to a version that does not have that specific
      option.)
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DecodeJpeg", name, contents, "channels", channels, "ratio",
        ratio, "fancy_upscaling", fancy_upscaling, "try_recover_truncated",
        try_recover_truncated, "acceptable_fraction", acceptable_fraction,
        "dct_method", dct_method)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return decode_jpeg_eager_fallback(
          contents, channels=channels, ratio=ratio,
          fancy_upscaling=fancy_upscaling,
          try_recover_truncated=try_recover_truncated,
          acceptable_fraction=acceptable_fraction, dct_method=dct_method,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if channels is None:
    channels = 0
  channels = _execute.make_int(channels, "channels")
  if ratio is None:
    ratio = 1
  ratio = _execute.make_int(ratio, "ratio")
  if fancy_upscaling is None:
    fancy_upscaling = True
  fancy_upscaling = _execute.make_bool(fancy_upscaling, "fancy_upscaling")
  if try_recover_truncated is None:
    try_recover_truncated = False
  try_recover_truncated = _execute.make_bool(try_recover_truncated, "try_recover_truncated")
  if acceptable_fraction is None:
    acceptable_fraction = 1
  acceptable_fraction = _execute.make_float(acceptable_fraction, "acceptable_fraction")
  if dct_method is None:
    dct_method = ""
  dct_method = _execute.make_str(dct_method, "dct_method")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DecodeJpeg", contents=contents, channels=channels, ratio=ratio,
                      fancy_upscaling=fancy_upscaling,
                      try_recover_truncated=try_recover_truncated,
                      acceptable_fraction=acceptable_fraction,
                      dct_method=dct_method, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("channels", _op._get_attr_int("channels"), "ratio",
              _op._get_attr_int("ratio"), "fancy_upscaling",
              _op._get_attr_bool("fancy_upscaling"), "try_recover_truncated",
              _op._get_attr_bool("try_recover_truncated"),
              "acceptable_fraction", _op.get_attr("acceptable_fraction"),
              "dct_method", _op.get_attr("dct_method"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DecodeJpeg", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DecodeJpeg = tf_export("raw_ops.DecodeJpeg")(_ops.to_raw_op(decode_jpeg))


def decode_jpeg_eager_fallback(contents: Annotated[Any, _atypes.String], channels: int, ratio: int, fancy_upscaling: bool, try_recover_truncated: bool, acceptable_fraction: float, dct_method: str, name, ctx) -> Annotated[Any, _atypes.UInt8]:
  if channels is None:
    channels = 0
  channels = _execute.make_int(channels, "channels")
  if ratio is None:
    ratio = 1
  ratio = _execute.make_int(ratio, "ratio")
  if fancy_upscaling is None:
    fancy_upscaling = True
  fancy_upscaling = _execute.make_bool(fancy_upscaling, "fancy_upscaling")
  if try_recover_truncated is None:
    try_recover_truncated = False
  try_recover_truncated = _execute.make_bool(try_recover_truncated, "try_recover_truncated")
  if acceptable_fraction is None:
    acceptable_fraction = 1
  acceptable_fraction = _execute.make_float(acceptable_fraction, "acceptable_fraction")
  if dct_method is None:
    dct_method = ""
  dct_method = _execute.make_str(dct_method, "dct_method")
  contents = _ops.convert_to_tensor(contents, _dtypes.string)
  _inputs_flat = [contents]
  _attrs = ("channels", channels, "ratio", ratio, "fancy_upscaling",
  fancy_upscaling, "try_recover_truncated", try_recover_truncated,
  "acceptable_fraction", acceptable_fraction, "dct_method", dct_method)
  _result = _execute.execute(b"DecodeJpeg", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DecodeJpeg", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DecodePng_dtype = TypeVar("TV_DecodePng_dtype", _atypes.UInt16, _atypes.UInt8)

def decode_png(contents: Annotated[Any, _atypes.String], channels:int=0, dtype:TV_DecodePng_dtype=_dtypes.uint8, name=None) -> Annotated[Any, TV_DecodePng_dtype]:
  r"""Decode a PNG-encoded image to a uint8 or uint16 tensor.

  The attr `channels` indicates the desired number of color channels for the
  decoded image.

  Accepted values are:

  *   0: Use the number of channels in the PNG-encoded image.
  *   1: output a grayscale image.
  *   3: output an RGB image.
  *   4: output an RGBA image.

  If needed, the PNG-encoded image is transformed to match the requested number
  of color channels.

  This op also supports decoding JPEGs and non-animated GIFs since the interface
  is the same, though it is cleaner to use `tf.io.decode_image`.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The PNG-encoded image.
    channels: An optional `int`. Defaults to `0`.
      Number of color channels for the decoded image.
    dtype: An optional `tf.DType` from: `tf.uint8, tf.uint16`. Defaults to `tf.uint8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DecodePng", name, contents, "channels", channels, "dtype",
        dtype)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return decode_png_eager_fallback(
          contents, channels=channels, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if channels is None:
    channels = 0
  channels = _execute.make_int(channels, "channels")
  if dtype is None:
    dtype = _dtypes.uint8
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DecodePng", contents=contents, channels=channels, dtype=dtype,
                     name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("channels", _op._get_attr_int("channels"), "dtype",
              _op._get_attr_type("dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DecodePng", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DecodePng = tf_export("raw_ops.DecodePng")(_ops.to_raw_op(decode_png))


def decode_png_eager_fallback(contents: Annotated[Any, _atypes.String], channels: int, dtype: TV_DecodePng_dtype, name, ctx) -> Annotated[Any, TV_DecodePng_dtype]:
  if channels is None:
    channels = 0
  channels = _execute.make_int(channels, "channels")
  if dtype is None:
    dtype = _dtypes.uint8
  dtype = _execute.make_type(dtype, "dtype")
  contents = _ops.convert_to_tensor(contents, _dtypes.string)
  _inputs_flat = [contents]
  _attrs = ("channels", channels, "dtype", dtype)
  _result = _execute.execute(b"DecodePng", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DecodePng", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DrawBoundingBoxes_T = TypeVar("TV_DrawBoundingBoxes_T", _atypes.Float32, _atypes.Half)

def draw_bounding_boxes(images: Annotated[Any, TV_DrawBoundingBoxes_T], boxes: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, TV_DrawBoundingBoxes_T]:
  r"""Draw bounding boxes on a batch of images.

  Outputs a copy of `images` but draws on top of the pixels zero or more bounding
  boxes specified by the locations in `boxes`. The coordinates of the each
  bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`. The
  bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
  height of the underlying image.

  For example, if an image is 100 x 200 pixels (height x width) and the bounding
  box is `[0.1, 0.2, 0.5, 0.9]`, the upper-left and bottom-right coordinates of
  the bounding box will be `(40, 10)` to `(180, 50)` (in (x,y) coordinates).

  Parts of the bounding box may fall outside the image.

  Args:
    images: A `Tensor`. Must be one of the following types: `float32`, `half`.
      4-D with shape `[batch, height, width, depth]`. A batch of images.
    boxes: A `Tensor` of type `float32`.
      3-D with shape `[batch, num_bounding_boxes, 4]` containing bounding
      boxes.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DrawBoundingBoxes", name, images, boxes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return draw_bounding_boxes_eager_fallback(
          images, boxes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DrawBoundingBoxes", images=images, boxes=boxes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DrawBoundingBoxes", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DrawBoundingBoxes = tf_export("raw_ops.DrawBoundingBoxes")(_ops.to_raw_op(draw_bounding_boxes))


def draw_bounding_boxes_eager_fallback(images: Annotated[Any, TV_DrawBoundingBoxes_T], boxes: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, TV_DrawBoundingBoxes_T]:
  _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.float32, _dtypes.half, ], _dtypes.float32)
  boxes = _ops.convert_to_tensor(boxes, _dtypes.float32)
  _inputs_flat = [images, boxes]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"DrawBoundingBoxes", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DrawBoundingBoxes", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DrawBoundingBoxesV2_T = TypeVar("TV_DrawBoundingBoxesV2_T", _atypes.Float32, _atypes.Half)

def draw_bounding_boxes_v2(images: Annotated[Any, TV_DrawBoundingBoxesV2_T], boxes: Annotated[Any, _atypes.Float32], colors: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, TV_DrawBoundingBoxesV2_T]:
  r"""Draw bounding boxes on a batch of images.

  Outputs a copy of `images` but draws on top of the pixels zero or more bounding
  boxes specified by the locations in `boxes`. The coordinates of the each
  bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`. The
  bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
  height of the underlying image.

  For example, if an image is 100 x 200 pixels (height x width) and the bounding
  box is `[0.1, 0.2, 0.5, 0.9]`, the upper-left and bottom-right coordinates of
  the bounding box will be `(40, 10)` to `(100, 50)` (in (x,y) coordinates).

  Parts of the bounding box may fall outside the image.

  Args:
    images: A `Tensor`. Must be one of the following types: `float32`, `half`.
      4-D with shape `[batch, height, width, depth]`. A batch of images.
    boxes: A `Tensor` of type `float32`.
      3-D with shape `[batch, num_bounding_boxes, 4]` containing bounding
      boxes.
    colors: A `Tensor` of type `float32`.
      2-D. A list of RGBA colors to cycle through for the boxes.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DrawBoundingBoxesV2", name, images, boxes, colors)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return draw_bounding_boxes_v2_eager_fallback(
          images, boxes, colors, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DrawBoundingBoxesV2", images=images, boxes=boxes, colors=colors,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DrawBoundingBoxesV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DrawBoundingBoxesV2 = tf_export("raw_ops.DrawBoundingBoxesV2")(_ops.to_raw_op(draw_bounding_boxes_v2))


def draw_bounding_boxes_v2_eager_fallback(images: Annotated[Any, TV_DrawBoundingBoxesV2_T], boxes: Annotated[Any, _atypes.Float32], colors: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, TV_DrawBoundingBoxesV2_T]:
  _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.float32, _dtypes.half, ], _dtypes.float32)
  boxes = _ops.convert_to_tensor(boxes, _dtypes.float32)
  colors = _ops.convert_to_tensor(colors, _dtypes.float32)
  _inputs_flat = [images, boxes, colors]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"DrawBoundingBoxesV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DrawBoundingBoxesV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def encode_jpeg(image: Annotated[Any, _atypes.UInt8], format:str="", quality:int=95, progressive:bool=False, optimize_size:bool=False, chroma_downsampling:bool=True, density_unit:str="in", x_density:int=300, y_density:int=300, xmp_metadata:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""JPEG-encode an image.

  `image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.

  The attr `format` can be used to override the color format of the encoded
  output.  Values can be:

  *   `''`: Use a default format based on the number of channels in the image.
  *   `grayscale`: Output a grayscale JPEG image.  The `channels` dimension
      of `image` must be 1.
  *   `rgb`: Output an RGB JPEG image. The `channels` dimension
      of `image` must be 3.

  If `format` is not specified or is the empty string, a default format is picked
  in function of the number of channels in `image`:

  *   1: Output a grayscale image.
  *   3: Output an RGB image.

  Args:
    image: A `Tensor` of type `uint8`.
      3-D with shape `[height, width, channels]`.
    format: An optional `string` from: `"", "grayscale", "rgb"`. Defaults to `""`.
      Per pixel image format.
    quality: An optional `int`. Defaults to `95`.
      Quality of the compression from 0 to 100 (higher is better and slower).
    progressive: An optional `bool`. Defaults to `False`.
      If True, create a JPEG that loads progressively (coarse to fine).
    optimize_size: An optional `bool`. Defaults to `False`.
      If True, spend CPU/RAM to reduce size with no quality change.
    chroma_downsampling: An optional `bool`. Defaults to `True`.
      See http://en.wikipedia.org/wiki/Chroma_subsampling.
    density_unit: An optional `string` from: `"in", "cm"`. Defaults to `"in"`.
      Unit used to specify `x_density` and `y_density`:
      pixels per inch (`'in'`) or centimeter (`'cm'`).
    x_density: An optional `int`. Defaults to `300`.
      Horizontal pixels per density unit.
    y_density: An optional `int`. Defaults to `300`.
      Vertical pixels per density unit.
    xmp_metadata: An optional `string`. Defaults to `""`.
      If not empty, embed this XMP metadata in the image header.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EncodeJpeg", name, image, "format", format, "quality", quality,
        "progressive", progressive, "optimize_size", optimize_size,
        "chroma_downsampling", chroma_downsampling, "density_unit",
        density_unit, "x_density", x_density, "y_density", y_density,
        "xmp_metadata", xmp_metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return encode_jpeg_eager_fallback(
          image, format=format, quality=quality, progressive=progressive,
          optimize_size=optimize_size,
          chroma_downsampling=chroma_downsampling, density_unit=density_unit,
          x_density=x_density, y_density=y_density, xmp_metadata=xmp_metadata,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if format is None:
    format = ""
  format = _execute.make_str(format, "format")
  if quality is None:
    quality = 95
  quality = _execute.make_int(quality, "quality")
  if progressive is None:
    progressive = False
  progressive = _execute.make_bool(progressive, "progressive")
  if optimize_size is None:
    optimize_size = False
  optimize_size = _execute.make_bool(optimize_size, "optimize_size")
  if chroma_downsampling is None:
    chroma_downsampling = True
  chroma_downsampling = _execute.make_bool(chroma_downsampling, "chroma_downsampling")
  if density_unit is None:
    density_unit = "in"
  density_unit = _execute.make_str(density_unit, "density_unit")
  if x_density is None:
    x_density = 300
  x_density = _execute.make_int(x_density, "x_density")
  if y_density is None:
    y_density = 300
  y_density = _execute.make_int(y_density, "y_density")
  if xmp_metadata is None:
    xmp_metadata = ""
  xmp_metadata = _execute.make_str(xmp_metadata, "xmp_metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EncodeJpeg", image=image, format=format, quality=quality,
                      progressive=progressive, optimize_size=optimize_size,
                      chroma_downsampling=chroma_downsampling,
                      density_unit=density_unit, x_density=x_density,
                      y_density=y_density, xmp_metadata=xmp_metadata,
                      name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("format", _op.get_attr("format"), "quality",
              _op._get_attr_int("quality"), "progressive",
              _op._get_attr_bool("progressive"), "optimize_size",
              _op._get_attr_bool("optimize_size"), "chroma_downsampling",
              _op._get_attr_bool("chroma_downsampling"), "density_unit",
              _op.get_attr("density_unit"), "x_density",
              _op._get_attr_int("x_density"), "y_density",
              _op._get_attr_int("y_density"), "xmp_metadata",
              _op.get_attr("xmp_metadata"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "EncodeJpeg", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

EncodeJpeg = tf_export("raw_ops.EncodeJpeg")(_ops.to_raw_op(encode_jpeg))


def encode_jpeg_eager_fallback(image: Annotated[Any, _atypes.UInt8], format: str, quality: int, progressive: bool, optimize_size: bool, chroma_downsampling: bool, density_unit: str, x_density: int, y_density: int, xmp_metadata: str, name, ctx) -> Annotated[Any, _atypes.String]:
  if format is None:
    format = ""
  format = _execute.make_str(format, "format")
  if quality is None:
    quality = 95
  quality = _execute.make_int(quality, "quality")
  if progressive is None:
    progressive = False
  progressive = _execute.make_bool(progressive, "progressive")
  if optimize_size is None:
    optimize_size = False
  optimize_size = _execute.make_bool(optimize_size, "optimize_size")
  if chroma_downsampling is None:
    chroma_downsampling = True
  chroma_downsampling = _execute.make_bool(chroma_downsampling, "chroma_downsampling")
  if density_unit is None:
    density_unit = "in"
  density_unit = _execute.make_str(density_unit, "density_unit")
  if x_density is None:
    x_density = 300
  x_density = _execute.make_int(x_density, "x_density")
  if y_density is None:
    y_density = 300
  y_density = _execute.make_int(y_density, "y_density")
  if xmp_metadata is None:
    xmp_metadata = ""
  xmp_metadata = _execute.make_str(xmp_metadata, "xmp_metadata")
  image = _ops.convert_to_tensor(image, _dtypes.uint8)
  _inputs_flat = [image]
  _attrs = ("format", format, "quality", quality, "progressive", progressive,
  "optimize_size", optimize_size, "chroma_downsampling", chroma_downsampling,
  "density_unit", density_unit, "x_density", x_density, "y_density",
  y_density, "xmp_metadata", xmp_metadata)
  _result = _execute.execute(b"EncodeJpeg", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "EncodeJpeg", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def encode_jpeg_variable_quality(images: Annotated[Any, _atypes.UInt8], quality: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, _atypes.String]:
  r"""JPEG encode input image with provided compression quality.

  `image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.
  `quality` is an int32 jpeg compression quality value between 0 and 100.

  Args:
    images: A `Tensor` of type `uint8`. Images to adjust.  At least 3-D.
    quality: A `Tensor` of type `int32`. An int quality to encode to.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EncodeJpegVariableQuality", name, images, quality)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return encode_jpeg_variable_quality_eager_fallback(
          images, quality, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EncodeJpegVariableQuality", images=images, quality=quality,
                                     name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "EncodeJpegVariableQuality", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

EncodeJpegVariableQuality = tf_export("raw_ops.EncodeJpegVariableQuality")(_ops.to_raw_op(encode_jpeg_variable_quality))


def encode_jpeg_variable_quality_eager_fallback(images: Annotated[Any, _atypes.UInt8], quality: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, _atypes.String]:
  images = _ops.convert_to_tensor(images, _dtypes.uint8)
  quality = _ops.convert_to_tensor(quality, _dtypes.int32)
  _inputs_flat = [images, quality]
  _attrs = None
  _result = _execute.execute(b"EncodeJpegVariableQuality", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "EncodeJpegVariableQuality", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_EncodePng_T = TypeVar("TV_EncodePng_T", _atypes.UInt16, _atypes.UInt8)

def encode_png(image: Annotated[Any, TV_EncodePng_T], compression:int=-1, name=None) -> Annotated[Any, _atypes.String]:
  r"""PNG-encode an image.

  `image` is a 3-D uint8 or uint16 Tensor of shape `[height, width, channels]`
  where `channels` is:

  *   1: for grayscale.
  *   2: for grayscale + alpha.
  *   3: for RGB.
  *   4: for RGBA.

  The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
  default or a value from 0 to 9.  9 is the highest compression level, generating
  the smallest output, but is slower.

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `uint16`.
      3-D with shape `[height, width, channels]`.
    compression: An optional `int`. Defaults to `-1`. Compression level.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EncodePng", name, image, "compression", compression)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return encode_png_eager_fallback(
          image, compression=compression, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if compression is None:
    compression = -1
  compression = _execute.make_int(compression, "compression")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EncodePng", image=image, compression=compression, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("compression", _op._get_attr_int("compression"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "EncodePng", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

EncodePng = tf_export("raw_ops.EncodePng")(_ops.to_raw_op(encode_png))


def encode_png_eager_fallback(image: Annotated[Any, TV_EncodePng_T], compression: int, name, ctx) -> Annotated[Any, _atypes.String]:
  if compression is None:
    compression = -1
  compression = _execute.make_int(compression, "compression")
  _attr_T, (image,) = _execute.args_to_matching_eager([image], ctx, [_dtypes.uint8, _dtypes.uint16, ], _dtypes.uint8)
  _inputs_flat = [image]
  _attrs = ("compression", compression, "T", _attr_T)
  _result = _execute.execute(b"EncodePng", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "EncodePng", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def extract_glimpse(input: Annotated[Any, _atypes.Float32], size: Annotated[Any, _atypes.Int32], offsets: Annotated[Any, _atypes.Float32], centered:bool=True, normalized:bool=True, uniform_noise:bool=True, noise:str="uniform", name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Extracts a glimpse from the input tensor.

  Returns a set of windows called glimpses extracted at location
  `offsets` from the input tensor. If the windows only partially
  overlaps the inputs, the non overlapping areas will be filled with
  random noise.

  The result is a 4-D tensor of shape `[batch_size, glimpse_height,
  glimpse_width, channels]`. The channels and batch dimensions are the
  same as that of the input tensor. The height and width of the output
  windows are specified in the `size` parameter.

  The argument `normalized` and `centered` controls how the windows are built:

  * If the coordinates are normalized but not centered, 0.0 and 1.0
    correspond to the minimum and maximum of each height and width
    dimension.
  * If the coordinates are both normalized and centered, they range from
    -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper
    left corner, the lower right corner is located at (1.0, 1.0) and the
    center is at (0, 0).
  * If the coordinates are not normalized they are interpreted as
    numbers of pixels.

  Args:
    input: A `Tensor` of type `float32`.
      A 4-D float tensor of shape `[batch_size, height, width, channels]`.
    size: A `Tensor` of type `int32`.
      A 1-D tensor of 2 elements containing the size of the glimpses
      to extract.  The glimpse height must be specified first, following
      by the glimpse width.
    offsets: A `Tensor` of type `float32`.
      A 2-D integer tensor of shape `[batch_size, 2]` containing
      the y, x locations of the center of each window.
    centered: An optional `bool`. Defaults to `True`.
      indicates if the offset coordinates are centered relative to
      the image, in which case the (0, 0) offset is relative to the center
      of the input images. If false, the (0,0) offset corresponds to the
      upper left corner of the input images.
    normalized: An optional `bool`. Defaults to `True`.
      indicates if the offset coordinates are normalized.
    uniform_noise: An optional `bool`. Defaults to `True`.
      indicates if the noise should be generated using a
      uniform distribution or a Gaussian distribution.
    noise: An optional `string`. Defaults to `"uniform"`.
      indicates if the noise should `uniform`, `gaussian`, or
      `zero`. The default is `uniform` which means the noise type
      will be decided by `uniform_noise`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExtractGlimpse", name, input, size, offsets, "centered",
        centered, "normalized", normalized, "uniform_noise", uniform_noise,
        "noise", noise)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return extract_glimpse_eager_fallback(
          input, size, offsets, centered=centered, normalized=normalized,
          uniform_noise=uniform_noise, noise=noise, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if centered is None:
    centered = True
  centered = _execute.make_bool(centered, "centered")
  if normalized is None:
    normalized = True
  normalized = _execute.make_bool(normalized, "normalized")
  if uniform_noise is None:
    uniform_noise = True
  uniform_noise = _execute.make_bool(uniform_noise, "uniform_noise")
  if noise is None:
    noise = "uniform"
  noise = _execute.make_str(noise, "noise")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExtractGlimpse", input=input, size=size, offsets=offsets,
                          centered=centered, normalized=normalized,
                          uniform_noise=uniform_noise, noise=noise, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("centered", _op._get_attr_bool("centered"), "normalized",
              _op._get_attr_bool("normalized"), "uniform_noise",
              _op._get_attr_bool("uniform_noise"), "noise",
              _op.get_attr("noise"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExtractGlimpse", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExtractGlimpse = tf_export("raw_ops.ExtractGlimpse")(_ops.to_raw_op(extract_glimpse))


def extract_glimpse_eager_fallback(input: Annotated[Any, _atypes.Float32], size: Annotated[Any, _atypes.Int32], offsets: Annotated[Any, _atypes.Float32], centered: bool, normalized: bool, uniform_noise: bool, noise: str, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if centered is None:
    centered = True
  centered = _execute.make_bool(centered, "centered")
  if normalized is None:
    normalized = True
  normalized = _execute.make_bool(normalized, "normalized")
  if uniform_noise is None:
    uniform_noise = True
  uniform_noise = _execute.make_bool(uniform_noise, "uniform_noise")
  if noise is None:
    noise = "uniform"
  noise = _execute.make_str(noise, "noise")
  input = _ops.convert_to_tensor(input, _dtypes.float32)
  size = _ops.convert_to_tensor(size, _dtypes.int32)
  offsets = _ops.convert_to_tensor(offsets, _dtypes.float32)
  _inputs_flat = [input, size, offsets]
  _attrs = ("centered", centered, "normalized", normalized, "uniform_noise",
  uniform_noise, "noise", noise)
  _result = _execute.execute(b"ExtractGlimpse", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExtractGlimpse", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def extract_glimpse_v2(input: Annotated[Any, _atypes.Float32], size: Annotated[Any, _atypes.Int32], offsets: Annotated[Any, _atypes.Float32], centered:bool=True, normalized:bool=True, uniform_noise:bool=True, noise:str="uniform", name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Extracts a glimpse from the input tensor.

  Returns a set of windows called glimpses extracted at location
  `offsets` from the input tensor. If the windows only partially
  overlaps the inputs, the non overlapping areas will be filled with
  random noise.

  The result is a 4-D tensor of shape `[batch_size, glimpse_height,
  glimpse_width, channels]`. The channels and batch dimensions are the
  same as that of the input tensor. The height and width of the output
  windows are specified in the `size` parameter.

  The argument `normalized` and `centered` controls how the windows are built:

  * If the coordinates are normalized but not centered, 0.0 and 1.0
    correspond to the minimum and maximum of each height and width
    dimension.
  * If the coordinates are both normalized and centered, they range from
    -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper
    left corner, the lower right corner is located at (1.0, 1.0) and the
    center is at (0, 0).
  * If the coordinates are not normalized they are interpreted as
    numbers of pixels.

  Args:
    input: A `Tensor` of type `float32`.
      A 4-D float tensor of shape `[batch_size, height, width, channels]`.
    size: A `Tensor` of type `int32`.
      A 1-D tensor of 2 elements containing the size of the glimpses
      to extract.  The glimpse height must be specified first, following
      by the glimpse width.
    offsets: A `Tensor` of type `float32`.
      A 2-D integer tensor of shape `[batch_size, 2]` containing
      the y, x locations of the center of each window.
    centered: An optional `bool`. Defaults to `True`.
      indicates if the offset coordinates are centered relative to
      the image, in which case the (0, 0) offset is relative to the center
      of the input images. If false, the (0,0) offset corresponds to the
      upper left corner of the input images.
    normalized: An optional `bool`. Defaults to `True`.
      indicates if the offset coordinates are normalized.
    uniform_noise: An optional `bool`. Defaults to `True`.
      indicates if the noise should be generated using a
      uniform distribution or a Gaussian distribution.
    noise: An optional `string`. Defaults to `"uniform"`.
      indicates if the noise should `uniform`, `gaussian`, or
      `zero`. The default is `uniform` which means the noise type
      will be decided by `uniform_noise`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExtractGlimpseV2", name, input, size, offsets, "centered",
        centered, "normalized", normalized, "uniform_noise", uniform_noise,
        "noise", noise)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return extract_glimpse_v2_eager_fallback(
          input, size, offsets, centered=centered, normalized=normalized,
          uniform_noise=uniform_noise, noise=noise, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if centered is None:
    centered = True
  centered = _execute.make_bool(centered, "centered")
  if normalized is None:
    normalized = True
  normalized = _execute.make_bool(normalized, "normalized")
  if uniform_noise is None:
    uniform_noise = True
  uniform_noise = _execute.make_bool(uniform_noise, "uniform_noise")
  if noise is None:
    noise = "uniform"
  noise = _execute.make_str(noise, "noise")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExtractGlimpseV2", input=input, size=size, offsets=offsets,
                            centered=centered, normalized=normalized,
                            uniform_noise=uniform_noise, noise=noise,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("centered", _op._get_attr_bool("centered"), "normalized",
              _op._get_attr_bool("normalized"), "uniform_noise",
              _op._get_attr_bool("uniform_noise"), "noise",
              _op.get_attr("noise"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExtractGlimpseV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExtractGlimpseV2 = tf_export("raw_ops.ExtractGlimpseV2")(_ops.to_raw_op(extract_glimpse_v2))


def extract_glimpse_v2_eager_fallback(input: Annotated[Any, _atypes.Float32], size: Annotated[Any, _atypes.Int32], offsets: Annotated[Any, _atypes.Float32], centered: bool, normalized: bool, uniform_noise: bool, noise: str, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if centered is None:
    centered = True
  centered = _execute.make_bool(centered, "centered")
  if normalized is None:
    normalized = True
  normalized = _execute.make_bool(normalized, "normalized")
  if uniform_noise is None:
    uniform_noise = True
  uniform_noise = _execute.make_bool(uniform_noise, "uniform_noise")
  if noise is None:
    noise = "uniform"
  noise = _execute.make_str(noise, "noise")
  input = _ops.convert_to_tensor(input, _dtypes.float32)
  size = _ops.convert_to_tensor(size, _dtypes.int32)
  offsets = _ops.convert_to_tensor(offsets, _dtypes.float32)
  _inputs_flat = [input, size, offsets]
  _attrs = ("centered", centered, "normalized", normalized, "uniform_noise",
  uniform_noise, "noise", noise)
  _result = _execute.execute(b"ExtractGlimpseV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExtractGlimpseV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ExtractJpegShape_output_type = TypeVar("TV_ExtractJpegShape_output_type", _atypes.Int32, _atypes.Int64)

def extract_jpeg_shape(contents: Annotated[Any, _atypes.String], output_type:TV_ExtractJpegShape_output_type=_dtypes.int32, name=None) -> Annotated[Any, TV_ExtractJpegShape_output_type]:
  r"""Extract the shape information of a JPEG-encoded image.

  This op only parses the image header, so it is much faster than DecodeJpeg.

  Args:
    contents: A `Tensor` of type `string`. 0-D. The JPEG-encoded image.
    output_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
      (Optional) The output type of the operation (int32 or int64).
      Defaults to int32.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExtractJpegShape", name, contents, "output_type", output_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return extract_jpeg_shape_eager_fallback(
          contents, output_type=output_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if output_type is None:
    output_type = _dtypes.int32
  output_type = _execute.make_type(output_type, "output_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExtractJpegShape", contents=contents, output_type=output_type,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_type", _op._get_attr_type("output_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExtractJpegShape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExtractJpegShape = tf_export("raw_ops.ExtractJpegShape")(_ops.to_raw_op(extract_jpeg_shape))


def extract_jpeg_shape_eager_fallback(contents: Annotated[Any, _atypes.String], output_type: TV_ExtractJpegShape_output_type, name, ctx) -> Annotated[Any, TV_ExtractJpegShape_output_type]:
  if output_type is None:
    output_type = _dtypes.int32
  output_type = _execute.make_type(output_type, "output_type")
  contents = _ops.convert_to_tensor(contents, _dtypes.string)
  _inputs_flat = [contents]
  _attrs = ("output_type", output_type)
  _result = _execute.execute(b"ExtractJpegShape", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExtractJpegShape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_GenerateBoundingBoxProposalsOutput = collections.namedtuple(
    "GenerateBoundingBoxProposals",
    ["rois", "roi_probabilities"])


def generate_bounding_box_proposals(scores: Annotated[Any, _atypes.Float32], bbox_deltas: Annotated[Any, _atypes.Float32], image_info: Annotated[Any, _atypes.Float32], anchors: Annotated[Any, _atypes.Float32], nms_threshold: Annotated[Any, _atypes.Float32], pre_nms_topn: Annotated[Any, _atypes.Int32], min_size: Annotated[Any, _atypes.Float32], post_nms_topn:int=300, name=None):
  r"""This op produces Region of Interests from given bounding boxes(bbox_deltas) encoded wrt anchors according to eq.2 in arXiv:1506.01497

        The op selects top `pre_nms_topn` scoring boxes, decodes them with respect to anchors,
        applies non-maximal suppression on overlapping boxes with higher than
        `nms_threshold` intersection-over-union (iou) value, discarding boxes where shorter
        side is less than `min_size`.
        Inputs:
        `scores`: A 4D tensor of shape [Batch, Height, Width, Num Anchors] containing the scores per anchor at given position
        `bbox_deltas`: is a tensor of shape [Batch, Height, Width, 4 x Num Anchors] boxes encoded to each anchor
        `anchors`: A 1D tensor of shape [4 x Num Anchors], representing the anchors.
        Outputs:
        `rois`: output RoIs, a 3D tensor of shape [Batch, post_nms_topn, 4], padded by 0 if less than post_nms_topn candidates found.
        `roi_probabilities`: probability scores of each roi in 'rois', a 2D tensor of shape [Batch,post_nms_topn], padded with 0 if needed, sorted by scores.

  Args:
    scores: A `Tensor` of type `float32`.
      A 4-D float tensor of shape `[num_images, height, width, num_achors]` containing scores of the boxes for given anchors, can be unsorted.
    bbox_deltas: A `Tensor` of type `float32`.
      A 4-D float tensor of shape `[num_images, height, width, 4 x num_anchors]`. encoding boxes with respec to each anchor.
      Coordinates are given in the form [dy, dx, dh, dw].
    image_info: A `Tensor` of type `float32`.
      A 2-D float tensor of shape `[num_images, 5]` containing image information Height, Width, Scale.
    anchors: A `Tensor` of type `float32`.
      A 2-D float tensor of shape `[num_anchors, 4]` describing the anchor boxes. Boxes are formatted in the form [y1, x1, y2, x2].
    nms_threshold: A `Tensor` of type `float32`.
      A scalar float tensor for non-maximal-suppression threshold.
    pre_nms_topn: A `Tensor` of type `int32`.
      A scalar int tensor for the number of top scoring boxes to be used as input.
    min_size: A `Tensor` of type `float32`.
      A scalar float tensor. Any box that has a smaller size than min_size will be discarded.
    post_nms_topn: An optional `int`. Defaults to `300`.
      An integer. Maximum number of rois in the output.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (rois, roi_probabilities).

    rois: A `Tensor` of type `float32`.
    roi_probabilities: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GenerateBoundingBoxProposals", name, scores, bbox_deltas,
        image_info, anchors, nms_threshold, pre_nms_topn, min_size,
        "post_nms_topn", post_nms_topn)
      _result = _GenerateBoundingBoxProposalsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return generate_bounding_box_proposals_eager_fallback(
          scores, bbox_deltas, image_info, anchors, nms_threshold,
          pre_nms_topn, min_size, post_nms_topn=post_nms_topn, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if post_nms_topn is None:
    post_nms_topn = 300
  post_nms_topn = _execute.make_int(post_nms_topn, "post_nms_topn")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GenerateBoundingBoxProposals", scores=scores,
                                        bbox_deltas=bbox_deltas,
                                        image_info=image_info,
                                        anchors=anchors,
                                        nms_threshold=nms_threshold,
                                        pre_nms_topn=pre_nms_topn,
                                        min_size=min_size,
                                        post_nms_topn=post_nms_topn,
                                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("post_nms_topn", _op._get_attr_int("post_nms_topn"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GenerateBoundingBoxProposals", _inputs_flat, _attrs, _result)
  _result = _GenerateBoundingBoxProposalsOutput._make(_result)
  return _result

GenerateBoundingBoxProposals = tf_export("raw_ops.GenerateBoundingBoxProposals")(_ops.to_raw_op(generate_bounding_box_proposals))


def generate_bounding_box_proposals_eager_fallback(scores: Annotated[Any, _atypes.Float32], bbox_deltas: Annotated[Any, _atypes.Float32], image_info: Annotated[Any, _atypes.Float32], anchors: Annotated[Any, _atypes.Float32], nms_threshold: Annotated[Any, _atypes.Float32], pre_nms_topn: Annotated[Any, _atypes.Int32], min_size: Annotated[Any, _atypes.Float32], post_nms_topn: int, name, ctx):
  if post_nms_topn is None:
    post_nms_topn = 300
  post_nms_topn = _execute.make_int(post_nms_topn, "post_nms_topn")
  scores = _ops.convert_to_tensor(scores, _dtypes.float32)
  bbox_deltas = _ops.convert_to_tensor(bbox_deltas, _dtypes.float32)
  image_info = _ops.convert_to_tensor(image_info, _dtypes.float32)
  anchors = _ops.convert_to_tensor(anchors, _dtypes.float32)
  nms_threshold = _ops.convert_to_tensor(nms_threshold, _dtypes.float32)
  pre_nms_topn = _ops.convert_to_tensor(pre_nms_topn, _dtypes.int32)
  min_size = _ops.convert_to_tensor(min_size, _dtypes.float32)
  _inputs_flat = [scores, bbox_deltas, image_info, anchors, nms_threshold, pre_nms_topn, min_size]
  _attrs = ("post_nms_topn", post_nms_topn)
  _result = _execute.execute(b"GenerateBoundingBoxProposals", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GenerateBoundingBoxProposals", _inputs_flat, _attrs, _result)
  _result = _GenerateBoundingBoxProposalsOutput._make(_result)
  return _result


TV_HSVToRGB_T = TypeVar("TV_HSVToRGB_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('image.hsv_to_rgb')
def hsv_to_rgb(images: Annotated[Any, TV_HSVToRGB_T], name=None) -> Annotated[Any, TV_HSVToRGB_T]:
  r"""Convert one or more images from HSV to RGB.

  Outputs a tensor of the same shape as the `images` tensor, containing the RGB
  value of the pixels. The output is only well defined if the value in `images`
  are in `[0,1]`.

  See `rgb_to_hsv` for a description of the HSV encoding.

  Args:
    images: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      1-D or higher rank. HSV data to convert. Last dimension must be size 3.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "HSVToRGB", name, images)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_hsv_to_rgb(
          (images, name,), None)
      if _result is not NotImplemented:
        return _result
      return hsv_to_rgb_eager_fallback(
          images, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            hsv_to_rgb, (), dict(images=images, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_hsv_to_rgb(
        (images, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "HSVToRGB", images=images, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          hsv_to_rgb, (), dict(images=images, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "HSVToRGB", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

HSVToRGB = tf_export("raw_ops.HSVToRGB")(_ops.to_raw_op(hsv_to_rgb))
_dispatcher_for_hsv_to_rgb = hsv_to_rgb._tf_type_based_dispatcher.Dispatch


def hsv_to_rgb_eager_fallback(images: Annotated[Any, TV_HSVToRGB_T], name, ctx) -> Annotated[Any, TV_HSVToRGB_T]:
  _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  _inputs_flat = [images]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"HSVToRGB", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "HSVToRGB", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ImageProjectiveTransformV2_dtype = TypeVar("TV_ImageProjectiveTransformV2_dtype", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64, _atypes.UInt8)

def image_projective_transform_v2(images: Annotated[Any, TV_ImageProjectiveTransformV2_dtype], transforms: Annotated[Any, _atypes.Float32], output_shape: Annotated[Any, _atypes.Int32], interpolation: str, fill_mode:str="CONSTANT", name=None) -> Annotated[Any, TV_ImageProjectiveTransformV2_dtype]:
  r"""Applies the given transform to each of the images.

  If one row of `transforms` is `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps
  the *output* point `(x, y)` to a transformed *input* point
  `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
  `k = c0 x + c1 y + 1`. If the transformed point lays outside of the input
  image, the output pixel is set to 0.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int32`, `int64`, `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    transforms: A `Tensor` of type `float32`.
      2-D Tensor, `[batch, 8]` or `[1, 8]` matrix, where each row corresponds to a 3 x 3
      projective transformation matrix, with the last entry assumed to be 1. If there
      is one row, the same transformation will be applied to all images.
    output_shape: A `Tensor` of type `int32`.
      1-D Tensor [new_height, new_width].
    interpolation: A `string`. Interpolation method, "NEAREST" or "BILINEAR".
    fill_mode: An optional `string`. Defaults to `"CONSTANT"`.
      Fill mode, "REFLECT", "WRAP", or "CONSTANT".
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ImageProjectiveTransformV2", name, images, transforms,
        output_shape, "interpolation", interpolation, "fill_mode", fill_mode)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return image_projective_transform_v2_eager_fallback(
          images, transforms, output_shape, interpolation=interpolation,
          fill_mode=fill_mode, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  interpolation = _execute.make_str(interpolation, "interpolation")
  if fill_mode is None:
    fill_mode = "CONSTANT"
  fill_mode = _execute.make_str(fill_mode, "fill_mode")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ImageProjectiveTransformV2", images=images, transforms=transforms,
                                      output_shape=output_shape,
                                      interpolation=interpolation,
                                      fill_mode=fill_mode, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "interpolation",
              _op.get_attr("interpolation"), "fill_mode",
              _op.get_attr("fill_mode"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ImageProjectiveTransformV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ImageProjectiveTransformV2 = tf_export("raw_ops.ImageProjectiveTransformV2")(_ops.to_raw_op(image_projective_transform_v2))


def image_projective_transform_v2_eager_fallback(images: Annotated[Any, TV_ImageProjectiveTransformV2_dtype], transforms: Annotated[Any, _atypes.Float32], output_shape: Annotated[Any, _atypes.Int32], interpolation: str, fill_mode: str, name, ctx) -> Annotated[Any, TV_ImageProjectiveTransformV2_dtype]:
  interpolation = _execute.make_str(interpolation, "interpolation")
  if fill_mode is None:
    fill_mode = "CONSTANT"
  fill_mode = _execute.make_str(fill_mode, "fill_mode")
  _attr_dtype, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.uint8, _dtypes.int32, _dtypes.int64, _dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  transforms = _ops.convert_to_tensor(transforms, _dtypes.float32)
  output_shape = _ops.convert_to_tensor(output_shape, _dtypes.int32)
  _inputs_flat = [images, transforms, output_shape]
  _attrs = ("dtype", _attr_dtype, "interpolation", interpolation, "fill_mode",
  fill_mode)
  _result = _execute.execute(b"ImageProjectiveTransformV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ImageProjectiveTransformV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ImageProjectiveTransformV3_dtype = TypeVar("TV_ImageProjectiveTransformV3_dtype", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64, _atypes.UInt8)

def image_projective_transform_v3(images: Annotated[Any, TV_ImageProjectiveTransformV3_dtype], transforms: Annotated[Any, _atypes.Float32], output_shape: Annotated[Any, _atypes.Int32], fill_value: Annotated[Any, _atypes.Float32], interpolation: str, fill_mode:str="CONSTANT", name=None) -> Annotated[Any, TV_ImageProjectiveTransformV3_dtype]:
  r"""Applies the given transform to each of the images.

  If one row of `transforms` is `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps
  the *output* point `(x, y)` to a transformed *input* point
  `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
  `k = c0 x + c1 y + 1`. If the transformed point lays outside of the input
  image, the output pixel is set to fill_value.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int32`, `int64`, `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    transforms: A `Tensor` of type `float32`.
      2-D Tensor, `[batch, 8]` or `[1, 8]` matrix, where each row corresponds to a 3 x 3
      projective transformation matrix, with the last entry assumed to be 1. If there
      is one row, the same transformation will be applied to all images.
    output_shape: A `Tensor` of type `int32`.
      1-D Tensor [new_height, new_width].
    fill_value: A `Tensor` of type `float32`.
      float, the value to be filled when fill_mode is constant".
    interpolation: A `string`. Interpolation method, "NEAREST" or "BILINEAR".
    fill_mode: An optional `string`. Defaults to `"CONSTANT"`.
      Fill mode, "REFLECT", "WRAP", "CONSTANT", or "NEAREST".
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ImageProjectiveTransformV3", name, images, transforms,
        output_shape, fill_value, "interpolation", interpolation, "fill_mode",
        fill_mode)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return image_projective_transform_v3_eager_fallback(
          images, transforms, output_shape, fill_value,
          interpolation=interpolation, fill_mode=fill_mode, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  interpolation = _execute.make_str(interpolation, "interpolation")
  if fill_mode is None:
    fill_mode = "CONSTANT"
  fill_mode = _execute.make_str(fill_mode, "fill_mode")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ImageProjectiveTransformV3", images=images, transforms=transforms,
                                      output_shape=output_shape,
                                      fill_value=fill_value,
                                      interpolation=interpolation,
                                      fill_mode=fill_mode, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "interpolation",
              _op.get_attr("interpolation"), "fill_mode",
              _op.get_attr("fill_mode"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ImageProjectiveTransformV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ImageProjectiveTransformV3 = tf_export("raw_ops.ImageProjectiveTransformV3")(_ops.to_raw_op(image_projective_transform_v3))


def image_projective_transform_v3_eager_fallback(images: Annotated[Any, TV_ImageProjectiveTransformV3_dtype], transforms: Annotated[Any, _atypes.Float32], output_shape: Annotated[Any, _atypes.Int32], fill_value: Annotated[Any, _atypes.Float32], interpolation: str, fill_mode: str, name, ctx) -> Annotated[Any, TV_ImageProjectiveTransformV3_dtype]:
  interpolation = _execute.make_str(interpolation, "interpolation")
  if fill_mode is None:
    fill_mode = "CONSTANT"
  fill_mode = _execute.make_str(fill_mode, "fill_mode")
  _attr_dtype, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.uint8, _dtypes.int32, _dtypes.int64, _dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  transforms = _ops.convert_to_tensor(transforms, _dtypes.float32)
  output_shape = _ops.convert_to_tensor(output_shape, _dtypes.int32)
  fill_value = _ops.convert_to_tensor(fill_value, _dtypes.float32)
  _inputs_flat = [images, transforms, output_shape, fill_value]
  _attrs = ("dtype", _attr_dtype, "interpolation", interpolation, "fill_mode",
  fill_mode)
  _result = _execute.execute(b"ImageProjectiveTransformV3", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ImageProjectiveTransformV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def non_max_suppression(boxes: Annotated[Any, _atypes.Float32], scores: Annotated[Any, _atypes.Float32], max_output_size: Annotated[Any, _atypes.Int32], iou_threshold:float=0.5, name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Greedily selects a subset of bounding boxes in descending order of score,

  pruning away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes are supplied as
  [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system.  Note that this
  algorithm is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather operation`.  For example:
    selected_indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size, iou_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)

  Args:
    boxes: A `Tensor` of type `float32`.
      A 2-D float tensor of shape `[num_boxes, 4]`.
    scores: A `Tensor` of type `float32`.
      A 1-D float tensor of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A `Tensor` of type `int32`.
      A scalar integer tensor representing the maximum number of
      boxes to be selected by non max suppression.
    iou_threshold: An optional `float`. Defaults to `0.5`.
      A float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NonMaxSuppression", name, boxes, scores, max_output_size,
        "iou_threshold", iou_threshold)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return non_max_suppression_eager_fallback(
          boxes, scores, max_output_size, iou_threshold=iou_threshold,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if iou_threshold is None:
    iou_threshold = 0.5
  iou_threshold = _execute.make_float(iou_threshold, "iou_threshold")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NonMaxSuppression", boxes=boxes, scores=scores,
                             max_output_size=max_output_size,
                             iou_threshold=iou_threshold, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("iou_threshold", _op.get_attr("iou_threshold"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NonMaxSuppression", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

NonMaxSuppression = tf_export("raw_ops.NonMaxSuppression")(_ops.to_raw_op(non_max_suppression))


def non_max_suppression_eager_fallback(boxes: Annotated[Any, _atypes.Float32], scores: Annotated[Any, _atypes.Float32], max_output_size: Annotated[Any, _atypes.Int32], iou_threshold: float, name, ctx) -> Annotated[Any, _atypes.Int32]:
  if iou_threshold is None:
    iou_threshold = 0.5
  iou_threshold = _execute.make_float(iou_threshold, "iou_threshold")
  boxes = _ops.convert_to_tensor(boxes, _dtypes.float32)
  scores = _ops.convert_to_tensor(scores, _dtypes.float32)
  max_output_size = _ops.convert_to_tensor(max_output_size, _dtypes.int32)
  _inputs_flat = [boxes, scores, max_output_size]
  _attrs = ("iou_threshold", iou_threshold)
  _result = _execute.execute(b"NonMaxSuppression", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NonMaxSuppression", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_NonMaxSuppressionV2_T = TypeVar("TV_NonMaxSuppressionV2_T", _atypes.Float32, _atypes.Half)
TV_NonMaxSuppressionV2_T_threshold = TypeVar("TV_NonMaxSuppressionV2_T_threshold", _atypes.Float32, _atypes.Half)

def non_max_suppression_v2(boxes: Annotated[Any, TV_NonMaxSuppressionV2_T], scores: Annotated[Any, TV_NonMaxSuppressionV2_T], max_output_size: Annotated[Any, _atypes.Int32], iou_threshold: Annotated[Any, TV_NonMaxSuppressionV2_T_threshold], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Greedily selects a subset of bounding boxes in descending order of score,

  pruning away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes are supplied as
  [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system.  Note that this
  algorithm is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.

  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather operation`.  For example:

    selected_indices = tf.image.non_max_suppression_v2(
        boxes, scores, max_output_size, iou_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)

  Args:
    boxes: A `Tensor`. Must be one of the following types: `half`, `float32`.
      A 2-D float tensor of shape `[num_boxes, 4]`.
    scores: A `Tensor`. Must have the same type as `boxes`.
      A 1-D float tensor of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A `Tensor` of type `int32`.
      A scalar integer tensor representing the maximum number of
      boxes to be selected by non max suppression.
    iou_threshold: A `Tensor`. Must be one of the following types: `half`, `float32`.
      A 0-D float tensor representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NonMaxSuppressionV2", name, boxes, scores, max_output_size,
        iou_threshold)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return non_max_suppression_v2_eager_fallback(
          boxes, scores, max_output_size, iou_threshold, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NonMaxSuppressionV2", boxes=boxes, scores=scores,
                               max_output_size=max_output_size,
                               iou_threshold=iou_threshold, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "T_threshold",
              _op._get_attr_type("T_threshold"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NonMaxSuppressionV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

NonMaxSuppressionV2 = tf_export("raw_ops.NonMaxSuppressionV2")(_ops.to_raw_op(non_max_suppression_v2))


def non_max_suppression_v2_eager_fallback(boxes: Annotated[Any, TV_NonMaxSuppressionV2_T], scores: Annotated[Any, TV_NonMaxSuppressionV2_T], max_output_size: Annotated[Any, _atypes.Int32], iou_threshold: Annotated[Any, TV_NonMaxSuppressionV2_T_threshold], name, ctx) -> Annotated[Any, _atypes.Int32]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([boxes, scores], ctx, [_dtypes.half, _dtypes.float32, ], _dtypes.float32)
  (boxes, scores) = _inputs_T
  _attr_T_threshold, (iou_threshold,) = _execute.args_to_matching_eager([iou_threshold], ctx, [_dtypes.half, _dtypes.float32, ], _dtypes.float32)
  max_output_size = _ops.convert_to_tensor(max_output_size, _dtypes.int32)
  _inputs_flat = [boxes, scores, max_output_size, iou_threshold]
  _attrs = ("T", _attr_T, "T_threshold", _attr_T_threshold)
  _result = _execute.execute(b"NonMaxSuppressionV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NonMaxSuppressionV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_NonMaxSuppressionV3_T = TypeVar("TV_NonMaxSuppressionV3_T", _atypes.Float32, _atypes.Half)
TV_NonMaxSuppressionV3_T_threshold = TypeVar("TV_NonMaxSuppressionV3_T_threshold", _atypes.Float32, _atypes.Half)

def non_max_suppression_v3(boxes: Annotated[Any, TV_NonMaxSuppressionV3_T], scores: Annotated[Any, TV_NonMaxSuppressionV3_T], max_output_size: Annotated[Any, _atypes.Int32], iou_threshold: Annotated[Any, TV_NonMaxSuppressionV3_T_threshold], score_threshold: Annotated[Any, TV_NonMaxSuppressionV3_T_threshold], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Greedily selects a subset of bounding boxes in descending order of score,

  pruning away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes with score less than
  `score_threshold` are removed.  Bounding boxes are supplied as
  [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system and more
  generally is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather operation`.  For example:
    selected_indices = tf.image.non_max_suppression_v2(
        boxes, scores, max_output_size, iou_threshold, score_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)

  Args:
    boxes: A `Tensor`. Must be one of the following types: `half`, `float32`.
      A 2-D float tensor of shape `[num_boxes, 4]`.
    scores: A `Tensor`. Must have the same type as `boxes`.
      A 1-D float tensor of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A `Tensor` of type `int32`.
      A scalar integer tensor representing the maximum number of
      boxes to be selected by non max suppression.
    iou_threshold: A `Tensor`. Must be one of the following types: `half`, `float32`.
      A 0-D float tensor representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    score_threshold: A `Tensor`. Must have the same type as `iou_threshold`.
      A 0-D float tensor representing the threshold for deciding when to remove
      boxes based on score.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NonMaxSuppressionV3", name, boxes, scores, max_output_size,
        iou_threshold, score_threshold)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return non_max_suppression_v3_eager_fallback(
          boxes, scores, max_output_size, iou_threshold, score_threshold,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NonMaxSuppressionV3", boxes=boxes, scores=scores,
                               max_output_size=max_output_size,
                               iou_threshold=iou_threshold,
                               score_threshold=score_threshold, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "T_threshold",
              _op._get_attr_type("T_threshold"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NonMaxSuppressionV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

NonMaxSuppressionV3 = tf_export("raw_ops.NonMaxSuppressionV3")(_ops.to_raw_op(non_max_suppression_v3))


def non_max_suppression_v3_eager_fallback(boxes: Annotated[Any, TV_NonMaxSuppressionV3_T], scores: Annotated[Any, TV_NonMaxSuppressionV3_T], max_output_size: Annotated[Any, _atypes.Int32], iou_threshold: Annotated[Any, TV_NonMaxSuppressionV3_T_threshold], score_threshold: Annotated[Any, TV_NonMaxSuppressionV3_T_threshold], name, ctx) -> Annotated[Any, _atypes.Int32]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([boxes, scores], ctx, [_dtypes.half, _dtypes.float32, ], _dtypes.float32)
  (boxes, scores) = _inputs_T
  _attr_T_threshold, _inputs_T_threshold = _execute.args_to_matching_eager([iou_threshold, score_threshold], ctx, [_dtypes.half, _dtypes.float32, ], _dtypes.float32)
  (iou_threshold, score_threshold) = _inputs_T_threshold
  max_output_size = _ops.convert_to_tensor(max_output_size, _dtypes.int32)
  _inputs_flat = [boxes, scores, max_output_size, iou_threshold, score_threshold]
  _attrs = ("T", _attr_T, "T_threshold", _attr_T_threshold)
  _result = _execute.execute(b"NonMaxSuppressionV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NonMaxSuppressionV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_NonMaxSuppressionV4Output = collections.namedtuple(
    "NonMaxSuppressionV4",
    ["selected_indices", "valid_outputs"])


TV_NonMaxSuppressionV4_T = TypeVar("TV_NonMaxSuppressionV4_T", _atypes.Float32, _atypes.Half)
TV_NonMaxSuppressionV4_T_threshold = TypeVar("TV_NonMaxSuppressionV4_T_threshold", _atypes.Float32, _atypes.Half)

def non_max_suppression_v4(boxes: Annotated[Any, TV_NonMaxSuppressionV4_T], scores: Annotated[Any, TV_NonMaxSuppressionV4_T], max_output_size: Annotated[Any, _atypes.Int32], iou_threshold: Annotated[Any, TV_NonMaxSuppressionV4_T_threshold], score_threshold: Annotated[Any, TV_NonMaxSuppressionV4_T_threshold], pad_to_max_output_size:bool=False, name=None):
  r"""Greedily selects a subset of bounding boxes in descending order of score,

  pruning away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes with score less than
  `score_threshold` are removed.  Bounding boxes are supplied as
  [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system and more
  generally is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather operation`.  For example:
    selected_indices = tf.image.non_max_suppression_v2(
        boxes, scores, max_output_size, iou_threshold, score_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)

  Args:
    boxes: A `Tensor`. Must be one of the following types: `half`, `float32`.
      A 2-D float tensor of shape `[num_boxes, 4]`.
    scores: A `Tensor`. Must have the same type as `boxes`.
      A 1-D float tensor of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A `Tensor` of type `int32`.
      A scalar integer tensor representing the maximum number of
      boxes to be selected by non max suppression.
    iou_threshold: A `Tensor`. Must be one of the following types: `half`, `float32`.
      A 0-D float tensor representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    score_threshold: A `Tensor`. Must have the same type as `iou_threshold`.
      A 0-D float tensor representing the threshold for deciding when to remove
      boxes based on score.
    pad_to_max_output_size: An optional `bool`. Defaults to `False`.
      If true, the output `selected_indices` is padded to be of length
      `max_output_size`. Defaults to false.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (selected_indices, valid_outputs).

    selected_indices: A `Tensor` of type `int32`.
    valid_outputs: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NonMaxSuppressionV4", name, boxes, scores, max_output_size,
        iou_threshold, score_threshold, "pad_to_max_output_size",
        pad_to_max_output_size)
      _result = _NonMaxSuppressionV4Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return non_max_suppression_v4_eager_fallback(
          boxes, scores, max_output_size, iou_threshold, score_threshold,
          pad_to_max_output_size=pad_to_max_output_size, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if pad_to_max_output_size is None:
    pad_to_max_output_size = False
  pad_to_max_output_size = _execute.make_bool(pad_to_max_output_size, "pad_to_max_output_size")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NonMaxSuppressionV4", boxes=boxes, scores=scores,
                               max_output_size=max_output_size,
                               iou_threshold=iou_threshold,
                               score_threshold=score_threshold,
                               pad_to_max_output_size=pad_to_max_output_size,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "T_threshold",
              _op._get_attr_type("T_threshold"), "pad_to_max_output_size",
              _op._get_attr_bool("pad_to_max_output_size"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NonMaxSuppressionV4", _inputs_flat, _attrs, _result)
  _result = _NonMaxSuppressionV4Output._make(_result)
  return _result

NonMaxSuppressionV4 = tf_export("raw_ops.NonMaxSuppressionV4")(_ops.to_raw_op(non_max_suppression_v4))


def non_max_suppression_v4_eager_fallback(boxes: Annotated[Any, TV_NonMaxSuppressionV4_T], scores: Annotated[Any, TV_NonMaxSuppressionV4_T], max_output_size: Annotated[Any, _atypes.Int32], iou_threshold: Annotated[Any, TV_NonMaxSuppressionV4_T_threshold], score_threshold: Annotated[Any, TV_NonMaxSuppressionV4_T_threshold], pad_to_max_output_size: bool, name, ctx):
  if pad_to_max_output_size is None:
    pad_to_max_output_size = False
  pad_to_max_output_size = _execute.make_bool(pad_to_max_output_size, "pad_to_max_output_size")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([boxes, scores], ctx, [_dtypes.half, _dtypes.float32, ], _dtypes.float32)
  (boxes, scores) = _inputs_T
  _attr_T_threshold, _inputs_T_threshold = _execute.args_to_matching_eager([iou_threshold, score_threshold], ctx, [_dtypes.half, _dtypes.float32, ], _dtypes.float32)
  (iou_threshold, score_threshold) = _inputs_T_threshold
  max_output_size = _ops.convert_to_tensor(max_output_size, _dtypes.int32)
  _inputs_flat = [boxes, scores, max_output_size, iou_threshold, score_threshold]
  _attrs = ("T", _attr_T, "T_threshold", _attr_T_threshold,
  "pad_to_max_output_size", pad_to_max_output_size)
  _result = _execute.execute(b"NonMaxSuppressionV4", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NonMaxSuppressionV4", _inputs_flat, _attrs, _result)
  _result = _NonMaxSuppressionV4Output._make(_result)
  return _result

_NonMaxSuppressionV5Output = collections.namedtuple(
    "NonMaxSuppressionV5",
    ["selected_indices", "selected_scores", "valid_outputs"])


TV_NonMaxSuppressionV5_T = TypeVar("TV_NonMaxSuppressionV5_T", _atypes.Float32, _atypes.Half)

def non_max_suppression_v5(boxes: Annotated[Any, TV_NonMaxSuppressionV5_T], scores: Annotated[Any, TV_NonMaxSuppressionV5_T], max_output_size: Annotated[Any, _atypes.Int32], iou_threshold: Annotated[Any, TV_NonMaxSuppressionV5_T], score_threshold: Annotated[Any, TV_NonMaxSuppressionV5_T], soft_nms_sigma: Annotated[Any, TV_NonMaxSuppressionV5_T], pad_to_max_output_size:bool=False, name=None):
  r"""Greedily selects a subset of bounding boxes in descending order of score,

  pruning away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes with score less than
  `score_threshold` are removed.  Bounding boxes are supplied as
  [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system and more
  generally is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather operation`.  For example:
    selected_indices = tf.image.non_max_suppression_v2(
        boxes, scores, max_output_size, iou_threshold, score_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)
  This op also supports a Soft-NMS (with Gaussian weighting) mode (c.f.
  Bodla et al, https://arxiv.org/abs/1704.04503) where boxes reduce the score
  of other overlapping boxes instead of directly causing them to be pruned.
  To enable this Soft-NMS mode, set the `soft_nms_sigma` parameter to be
  larger than 0.

  Args:
    boxes: A `Tensor`. Must be one of the following types: `half`, `float32`.
      A 2-D float tensor of shape `[num_boxes, 4]`.
    scores: A `Tensor`. Must have the same type as `boxes`.
      A 1-D float tensor of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A `Tensor` of type `int32`.
      A scalar integer tensor representing the maximum number of
      boxes to be selected by non max suppression.
    iou_threshold: A `Tensor`. Must have the same type as `boxes`.
      A 0-D float tensor representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    score_threshold: A `Tensor`. Must have the same type as `boxes`.
      A 0-D float tensor representing the threshold for deciding when to remove
      boxes based on score.
    soft_nms_sigma: A `Tensor`. Must have the same type as `boxes`.
      A 0-D float tensor representing the sigma parameter for Soft NMS; see Bodla et
      al (c.f. https://arxiv.org/abs/1704.04503).  When `soft_nms_sigma=0.0` (which
      is default), we fall back to standard (hard) NMS.
    pad_to_max_output_size: An optional `bool`. Defaults to `False`.
      If true, the output `selected_indices` is padded to be of length
      `max_output_size`. Defaults to false.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (selected_indices, selected_scores, valid_outputs).

    selected_indices: A `Tensor` of type `int32`.
    selected_scores: A `Tensor`. Has the same type as `boxes`.
    valid_outputs: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NonMaxSuppressionV5", name, boxes, scores, max_output_size,
        iou_threshold, score_threshold, soft_nms_sigma,
        "pad_to_max_output_size", pad_to_max_output_size)
      _result = _NonMaxSuppressionV5Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return non_max_suppression_v5_eager_fallback(
          boxes, scores, max_output_size, iou_threshold, score_threshold,
          soft_nms_sigma, pad_to_max_output_size=pad_to_max_output_size,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if pad_to_max_output_size is None:
    pad_to_max_output_size = False
  pad_to_max_output_size = _execute.make_bool(pad_to_max_output_size, "pad_to_max_output_size")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NonMaxSuppressionV5", boxes=boxes, scores=scores,
                               max_output_size=max_output_size,
                               iou_threshold=iou_threshold,
                               score_threshold=score_threshold,
                               soft_nms_sigma=soft_nms_sigma,
                               pad_to_max_output_size=pad_to_max_output_size,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "pad_to_max_output_size",
              _op._get_attr_bool("pad_to_max_output_size"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NonMaxSuppressionV5", _inputs_flat, _attrs, _result)
  _result = _NonMaxSuppressionV5Output._make(_result)
  return _result

NonMaxSuppressionV5 = tf_export("raw_ops.NonMaxSuppressionV5")(_ops.to_raw_op(non_max_suppression_v5))


def non_max_suppression_v5_eager_fallback(boxes: Annotated[Any, TV_NonMaxSuppressionV5_T], scores: Annotated[Any, TV_NonMaxSuppressionV5_T], max_output_size: Annotated[Any, _atypes.Int32], iou_threshold: Annotated[Any, TV_NonMaxSuppressionV5_T], score_threshold: Annotated[Any, TV_NonMaxSuppressionV5_T], soft_nms_sigma: Annotated[Any, TV_NonMaxSuppressionV5_T], pad_to_max_output_size: bool, name, ctx):
  if pad_to_max_output_size is None:
    pad_to_max_output_size = False
  pad_to_max_output_size = _execute.make_bool(pad_to_max_output_size, "pad_to_max_output_size")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([boxes, scores, iou_threshold, score_threshold, soft_nms_sigma], ctx, [_dtypes.half, _dtypes.float32, ], _dtypes.float32)
  (boxes, scores, iou_threshold, score_threshold, soft_nms_sigma) = _inputs_T
  max_output_size = _ops.convert_to_tensor(max_output_size, _dtypes.int32)
  _inputs_flat = [boxes, scores, max_output_size, iou_threshold, score_threshold, soft_nms_sigma]
  _attrs = ("T", _attr_T, "pad_to_max_output_size", pad_to_max_output_size)
  _result = _execute.execute(b"NonMaxSuppressionV5", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NonMaxSuppressionV5", _inputs_flat, _attrs, _result)
  _result = _NonMaxSuppressionV5Output._make(_result)
  return _result


def non_max_suppression_with_overlaps(overlaps: Annotated[Any, _atypes.Float32], scores: Annotated[Any, _atypes.Float32], max_output_size: Annotated[Any, _atypes.Int32], overlap_threshold: Annotated[Any, _atypes.Float32], score_threshold: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Greedily selects a subset of bounding boxes in descending order of score,

  pruning away boxes that have high overlaps
  with previously selected boxes.  Bounding boxes with score less than
  `score_threshold` are removed. N-by-n overlap values are supplied as square matrix,
  which allows for defining a custom overlap criterium (eg. intersection over union,
  intersection over area, etc.).

  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather operation`.  For example:

    selected_indices = tf.image.non_max_suppression_with_overlaps(
        overlaps, scores, max_output_size, overlap_threshold, score_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)

  Args:
    overlaps: A `Tensor` of type `float32`.
      A 2-D float tensor of shape `[num_boxes, num_boxes]` representing
      the n-by-n box overlap values.
    scores: A `Tensor` of type `float32`.
      A 1-D float tensor of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A `Tensor` of type `int32`.
      A scalar integer tensor representing the maximum number of
      boxes to be selected by non max suppression.
    overlap_threshold: A `Tensor` of type `float32`.
      A 0-D float tensor representing the threshold for deciding whether
      boxes overlap too.
    score_threshold: A `Tensor` of type `float32`.
      A 0-D float tensor representing the threshold for deciding when to remove
      boxes based on score.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NonMaxSuppressionWithOverlaps", name, overlaps, scores,
        max_output_size, overlap_threshold, score_threshold)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return non_max_suppression_with_overlaps_eager_fallback(
          overlaps, scores, max_output_size, overlap_threshold,
          score_threshold, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NonMaxSuppressionWithOverlaps", overlaps=overlaps, scores=scores,
                                         max_output_size=max_output_size,
                                         overlap_threshold=overlap_threshold,
                                         score_threshold=score_threshold,
                                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NonMaxSuppressionWithOverlaps", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

NonMaxSuppressionWithOverlaps = tf_export("raw_ops.NonMaxSuppressionWithOverlaps")(_ops.to_raw_op(non_max_suppression_with_overlaps))


def non_max_suppression_with_overlaps_eager_fallback(overlaps: Annotated[Any, _atypes.Float32], scores: Annotated[Any, _atypes.Float32], max_output_size: Annotated[Any, _atypes.Int32], overlap_threshold: Annotated[Any, _atypes.Float32], score_threshold: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Int32]:
  overlaps = _ops.convert_to_tensor(overlaps, _dtypes.float32)
  scores = _ops.convert_to_tensor(scores, _dtypes.float32)
  max_output_size = _ops.convert_to_tensor(max_output_size, _dtypes.int32)
  overlap_threshold = _ops.convert_to_tensor(overlap_threshold, _dtypes.float32)
  score_threshold = _ops.convert_to_tensor(score_threshold, _dtypes.float32)
  _inputs_flat = [overlaps, scores, max_output_size, overlap_threshold, score_threshold]
  _attrs = None
  _result = _execute.execute(b"NonMaxSuppressionWithOverlaps", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NonMaxSuppressionWithOverlaps", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_QuantizedResizeBilinearOutput = collections.namedtuple(
    "QuantizedResizeBilinear",
    ["resized_images", "out_min", "out_max"])


TV_QuantizedResizeBilinear_T = TypeVar("TV_QuantizedResizeBilinear_T", _atypes.Float32, _atypes.QInt32, _atypes.QUInt8)

def quantized_resize_bilinear(images: Annotated[Any, TV_QuantizedResizeBilinear_T], size: Annotated[Any, _atypes.Int32], min: Annotated[Any, _atypes.Float32], max: Annotated[Any, _atypes.Float32], align_corners:bool=False, half_pixel_centers:bool=False, name=None):
  r"""Resize quantized `images` to `size` using quantized bilinear interpolation.

  Input images and output images must be quantized types.

  Args:
    images: A `Tensor`. Must be one of the following types: `quint8`, `qint32`, `float32`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    min: A `Tensor` of type `float32`.
    max: A `Tensor` of type `float32`.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, the centers of the 4 corner pixels of the input and output tensors are
      aligned, preserving the values at the corner pixels. Defaults to false.
    half_pixel_centers: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (resized_images, out_min, out_max).

    resized_images: A `Tensor`. Has the same type as `images`.
    out_min: A `Tensor` of type `float32`.
    out_max: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedResizeBilinear", name, images, size, min, max,
        "align_corners", align_corners, "half_pixel_centers",
        half_pixel_centers)
      _result = _QuantizedResizeBilinearOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_resize_bilinear_eager_fallback(
          images, size, min, max, align_corners=align_corners,
          half_pixel_centers=half_pixel_centers, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if align_corners is None:
    align_corners = False
  align_corners = _execute.make_bool(align_corners, "align_corners")
  if half_pixel_centers is None:
    half_pixel_centers = False
  half_pixel_centers = _execute.make_bool(half_pixel_centers, "half_pixel_centers")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedResizeBilinear", images=images, size=size, min=min, max=max,
                                   align_corners=align_corners,
                                   half_pixel_centers=half_pixel_centers,
                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "align_corners",
              _op._get_attr_bool("align_corners"), "half_pixel_centers",
              _op._get_attr_bool("half_pixel_centers"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedResizeBilinear", _inputs_flat, _attrs, _result)
  _result = _QuantizedResizeBilinearOutput._make(_result)
  return _result

QuantizedResizeBilinear = tf_export("raw_ops.QuantizedResizeBilinear")(_ops.to_raw_op(quantized_resize_bilinear))


def quantized_resize_bilinear_eager_fallback(images: Annotated[Any, TV_QuantizedResizeBilinear_T], size: Annotated[Any, _atypes.Int32], min: Annotated[Any, _atypes.Float32], max: Annotated[Any, _atypes.Float32], align_corners: bool, half_pixel_centers: bool, name, ctx):
  if align_corners is None:
    align_corners = False
  align_corners = _execute.make_bool(align_corners, "align_corners")
  if half_pixel_centers is None:
    half_pixel_centers = False
  half_pixel_centers = _execute.make_bool(half_pixel_centers, "half_pixel_centers")
  _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.quint8, _dtypes.qint32, _dtypes.float32, ])
  size = _ops.convert_to_tensor(size, _dtypes.int32)
  min = _ops.convert_to_tensor(min, _dtypes.float32)
  max = _ops.convert_to_tensor(max, _dtypes.float32)
  _inputs_flat = [images, size, min, max]
  _attrs = ("T", _attr_T, "align_corners", align_corners,
  "half_pixel_centers", half_pixel_centers)
  _result = _execute.execute(b"QuantizedResizeBilinear", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedResizeBilinear", _inputs_flat, _attrs, _result)
  _result = _QuantizedResizeBilinearOutput._make(_result)
  return _result


TV_RGBToHSV_T = TypeVar("TV_RGBToHSV_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('image.rgb_to_hsv')
def rgb_to_hsv(images: Annotated[Any, TV_RGBToHSV_T], name=None) -> Annotated[Any, TV_RGBToHSV_T]:
  r"""Converts one or more images from RGB to HSV.

  Outputs a tensor of the same shape as the `images` tensor, containing the HSV
  value of the pixels. The output is only well defined if the value in `images`
  are in `[0,1]`.

  `output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
  `output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
  corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.

  Usage Example:

  >>> blue_image = tf.stack([
  ...    tf.zeros([5,5]),
  ...    tf.zeros([5,5]),
  ...    tf.ones([5,5])],
  ...    axis=-1)
  >>> blue_hsv_image = tf.image.rgb_to_hsv(blue_image)
  >>> blue_hsv_image[0,0].numpy()
  array([0.6666667, 1. , 1. ], dtype=float32)

  Args:
    images: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      1-D or higher rank. RGB data to convert. Last dimension must be size 3.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RGBToHSV", name, images)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_rgb_to_hsv(
          (images, name,), None)
      if _result is not NotImplemented:
        return _result
      return rgb_to_hsv_eager_fallback(
          images, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            rgb_to_hsv, (), dict(images=images, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_rgb_to_hsv(
        (images, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RGBToHSV", images=images, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          rgb_to_hsv, (), dict(images=images, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RGBToHSV", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RGBToHSV = tf_export("raw_ops.RGBToHSV")(_ops.to_raw_op(rgb_to_hsv))
_dispatcher_for_rgb_to_hsv = rgb_to_hsv._tf_type_based_dispatcher.Dispatch


def rgb_to_hsv_eager_fallback(images: Annotated[Any, TV_RGBToHSV_T], name, ctx) -> Annotated[Any, TV_RGBToHSV_T]:
  _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  _inputs_flat = [images]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"RGBToHSV", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RGBToHSV", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RandomCrop_T = TypeVar("TV_RandomCrop_T", _atypes.Float32, _atypes.Float64, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt8)

def random_crop(image: Annotated[Any, TV_RandomCrop_T], size: Annotated[Any, _atypes.Int64], seed:int=0, seed2:int=0, name=None) -> Annotated[Any, TV_RandomCrop_T]:
  r"""Randomly crop `image`.

  `size` is a 1-D int64 tensor with 2 elements representing the crop height and
  width.  The values must be non negative.

  This Op picks a random location in `image` and crops a `height` by `width`
  rectangle from that location.  The random location is picked so the cropped
  area will fit inside the original image.

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`.
      3-D of shape `[height, width, channels]`.
    size: A `Tensor` of type `int64`.
      1-D of length 2 containing: `crop_height`, `crop_width`..
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `image`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RandomCrop", name, image, size, "seed", seed, "seed2", seed2)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return random_crop_eager_fallback(
          image, size, seed=seed, seed2=seed2, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RandomCrop", image=image, size=size, seed=seed, seed2=seed2,
                      name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "seed", _op._get_attr_int("seed"),
              "seed2", _op._get_attr_int("seed2"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RandomCrop", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RandomCrop = tf_export("raw_ops.RandomCrop")(_ops.to_raw_op(random_crop))


def random_crop_eager_fallback(image: Annotated[Any, TV_RandomCrop_T], size: Annotated[Any, _atypes.Int64], seed: int, seed2: int, name, ctx) -> Annotated[Any, TV_RandomCrop_T]:
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _attr_T, (image,) = _execute.args_to_matching_eager([image], ctx, [_dtypes.uint8, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.float32, _dtypes.float64, ])
  size = _ops.convert_to_tensor(size, _dtypes.int64)
  _inputs_flat = [image, size]
  _attrs = ("T", _attr_T, "seed", seed, "seed2", seed2)
  _result = _execute.execute(b"RandomCrop", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RandomCrop", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ResizeArea_T = TypeVar("TV_ResizeArea_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt8)

def resize_area(images: Annotated[Any, TV_ResizeArea_T], size: Annotated[Any, _atypes.Int32], align_corners:bool=False, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Resize `images` to `size` using area interpolation.

  Input images can be of different types but output images are always float.

  The range of pixel values for the output image might be slightly different
  from the range for the input image because of limited numerical precision.
  To guarantee an output range, for example `[0.0, 1.0]`, apply
  `tf.clip_by_value` to the output.

  Each output pixel is computed by first transforming the pixel's footprint into
  the input tensor and then averaging the pixels that intersect the footprint. An
  input pixel's contribution to the average is weighted by the fraction of its
  area that intersects the footprint.  This is the same as OpenCV's INTER_AREA.

  Args:
    images: A `Tensor`. Must be one of the following types: `int8`, `uint8`, `int16`, `uint16`, `int32`, `int64`, `half`, `float32`, `float64`, `bfloat16`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, the centers of the 4 corner pixels of the input and output tensors are
      aligned, preserving the values at the corner pixels. Defaults to false.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResizeArea", name, images, size, "align_corners",
        align_corners)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resize_area_eager_fallback(
          images, size, align_corners=align_corners, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if align_corners is None:
    align_corners = False
  align_corners = _execute.make_bool(align_corners, "align_corners")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResizeArea", images=images, size=size, align_corners=align_corners,
                      name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "align_corners",
              _op._get_attr_bool("align_corners"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ResizeArea", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ResizeArea = tf_export("raw_ops.ResizeArea")(_ops.to_raw_op(resize_area))


def resize_area_eager_fallback(images: Annotated[Any, TV_ResizeArea_T], size: Annotated[Any, _atypes.Int32], align_corners: bool, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if align_corners is None:
    align_corners = False
  align_corners = _execute.make_bool(align_corners, "align_corners")
  _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.int8, _dtypes.uint8, _dtypes.int16, _dtypes.uint16, _dtypes.int32, _dtypes.int64, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.bfloat16, ])
  size = _ops.convert_to_tensor(size, _dtypes.int32)
  _inputs_flat = [images, size]
  _attrs = ("T", _attr_T, "align_corners", align_corners)
  _result = _execute.execute(b"ResizeArea", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ResizeArea", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ResizeBicubic_T = TypeVar("TV_ResizeBicubic_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt8)

def resize_bicubic(images: Annotated[Any, TV_ResizeBicubic_T], size: Annotated[Any, _atypes.Int32], align_corners:bool=False, half_pixel_centers:bool=False, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Resize `images` to `size` using bicubic interpolation.

  Input images can be of different types but output images are always float.

  Args:
    images: A `Tensor`. Must be one of the following types: `int8`, `uint8`, `int16`, `uint16`, `int32`, `int64`, `half`, `float32`, `float64`, `bfloat16`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, the centers of the 4 corner pixels of the input and output tensors are
      aligned, preserving the values at the corner pixels. Defaults to false.
    half_pixel_centers: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResizeBicubic", name, images, size, "align_corners",
        align_corners, "half_pixel_centers", half_pixel_centers)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resize_bicubic_eager_fallback(
          images, size, align_corners=align_corners,
          half_pixel_centers=half_pixel_centers, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if align_corners is None:
    align_corners = False
  align_corners = _execute.make_bool(align_corners, "align_corners")
  if half_pixel_centers is None:
    half_pixel_centers = False
  half_pixel_centers = _execute.make_bool(half_pixel_centers, "half_pixel_centers")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResizeBicubic", images=images, size=size,
                         align_corners=align_corners,
                         half_pixel_centers=half_pixel_centers, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "align_corners",
              _op._get_attr_bool("align_corners"), "half_pixel_centers",
              _op._get_attr_bool("half_pixel_centers"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ResizeBicubic", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ResizeBicubic = tf_export("raw_ops.ResizeBicubic")(_ops.to_raw_op(resize_bicubic))


def resize_bicubic_eager_fallback(images: Annotated[Any, TV_ResizeBicubic_T], size: Annotated[Any, _atypes.Int32], align_corners: bool, half_pixel_centers: bool, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if align_corners is None:
    align_corners = False
  align_corners = _execute.make_bool(align_corners, "align_corners")
  if half_pixel_centers is None:
    half_pixel_centers = False
  half_pixel_centers = _execute.make_bool(half_pixel_centers, "half_pixel_centers")
  _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.int8, _dtypes.uint8, _dtypes.int16, _dtypes.uint16, _dtypes.int32, _dtypes.int64, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.bfloat16, ])
  size = _ops.convert_to_tensor(size, _dtypes.int32)
  _inputs_flat = [images, size]
  _attrs = ("T", _attr_T, "align_corners", align_corners,
  "half_pixel_centers", half_pixel_centers)
  _result = _execute.execute(b"ResizeBicubic", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ResizeBicubic", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ResizeBicubicGrad_T = TypeVar("TV_ResizeBicubicGrad_T", _atypes.Float32, _atypes.Float64)

def resize_bicubic_grad(grads: Annotated[Any, _atypes.Float32], original_image: Annotated[Any, TV_ResizeBicubicGrad_T], align_corners:bool=False, half_pixel_centers:bool=False, name=None) -> Annotated[Any, TV_ResizeBicubicGrad_T]:
  r"""Computes the gradient of bicubic interpolation.

  Args:
    grads: A `Tensor` of type `float32`.
      4-D with shape `[batch, height, width, channels]`.
    original_image: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      4-D with shape `[batch, orig_height, orig_width, channels]`,
      The image tensor that was resized.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, the centers of the 4 corner pixels of the input and grad tensors are
      aligned. Defaults to false.
    half_pixel_centers: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `original_image`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResizeBicubicGrad", name, grads, original_image,
        "align_corners", align_corners, "half_pixel_centers",
        half_pixel_centers)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resize_bicubic_grad_eager_fallback(
          grads, original_image, align_corners=align_corners,
          half_pixel_centers=half_pixel_centers, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if align_corners is None:
    align_corners = False
  align_corners = _execute.make_bool(align_corners, "align_corners")
  if half_pixel_centers is None:
    half_pixel_centers = False
  half_pixel_centers = _execute.make_bool(half_pixel_centers, "half_pixel_centers")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResizeBicubicGrad", grads=grads, original_image=original_image,
                             align_corners=align_corners,
                             half_pixel_centers=half_pixel_centers, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "align_corners",
              _op._get_attr_bool("align_corners"), "half_pixel_centers",
              _op._get_attr_bool("half_pixel_centers"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ResizeBicubicGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ResizeBicubicGrad = tf_export("raw_ops.ResizeBicubicGrad")(_ops.to_raw_op(resize_bicubic_grad))


def resize_bicubic_grad_eager_fallback(grads: Annotated[Any, _atypes.Float32], original_image: Annotated[Any, TV_ResizeBicubicGrad_T], align_corners: bool, half_pixel_centers: bool, name, ctx) -> Annotated[Any, TV_ResizeBicubicGrad_T]:
  if align_corners is None:
    align_corners = False
  align_corners = _execute.make_bool(align_corners, "align_corners")
  if half_pixel_centers is None:
    half_pixel_centers = False
  half_pixel_centers = _execute.make_bool(half_pixel_centers, "half_pixel_centers")
  _attr_T, (original_image,) = _execute.args_to_matching_eager([original_image], ctx, [_dtypes.float32, _dtypes.float64, ])
  grads = _ops.convert_to_tensor(grads, _dtypes.float32)
  _inputs_flat = [grads, original_image]
  _attrs = ("T", _attr_T, "align_corners", align_corners,
  "half_pixel_centers", half_pixel_centers)
  _result = _execute.execute(b"ResizeBicubicGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ResizeBicubicGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ResizeBilinear_T = TypeVar("TV_ResizeBilinear_T", _atypes.BFloat16, _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt8)

def resize_bilinear(images: Annotated[Any, TV_ResizeBilinear_T], size: Annotated[Any, _atypes.Int32], align_corners:bool=False, half_pixel_centers:bool=False, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Resize `images` to `size` using bilinear interpolation.

  Input images can be of different types but output images are always float.

  Args:
    images: A `Tensor`. Must be one of the following types: `int8`, `uint8`, `int16`, `uint16`, `int32`, `int64`, `bfloat16`, `half`, `float32`, `float64`, `bfloat16`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, the centers of the 4 corner pixels of the input and output tensors are
      aligned, preserving the values at the corner pixels. Defaults to false.
    half_pixel_centers: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResizeBilinear", name, images, size, "align_corners",
        align_corners, "half_pixel_centers", half_pixel_centers)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resize_bilinear_eager_fallback(
          images, size, align_corners=align_corners,
          half_pixel_centers=half_pixel_centers, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if align_corners is None:
    align_corners = False
  align_corners = _execute.make_bool(align_corners, "align_corners")
  if half_pixel_centers is None:
    half_pixel_centers = False
  half_pixel_centers = _execute.make_bool(half_pixel_centers, "half_pixel_centers")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResizeBilinear", images=images, size=size,
                          align_corners=align_corners,
                          half_pixel_centers=half_pixel_centers, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "align_corners",
              _op._get_attr_bool("align_corners"), "half_pixel_centers",
              _op._get_attr_bool("half_pixel_centers"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ResizeBilinear", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ResizeBilinear = tf_export("raw_ops.ResizeBilinear")(_ops.to_raw_op(resize_bilinear))


def resize_bilinear_eager_fallback(images: Annotated[Any, TV_ResizeBilinear_T], size: Annotated[Any, _atypes.Int32], align_corners: bool, half_pixel_centers: bool, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if align_corners is None:
    align_corners = False
  align_corners = _execute.make_bool(align_corners, "align_corners")
  if half_pixel_centers is None:
    half_pixel_centers = False
  half_pixel_centers = _execute.make_bool(half_pixel_centers, "half_pixel_centers")
  _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.int8, _dtypes.uint8, _dtypes.int16, _dtypes.uint16, _dtypes.int32, _dtypes.int64, _dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.bfloat16, ])
  size = _ops.convert_to_tensor(size, _dtypes.int32)
  _inputs_flat = [images, size]
  _attrs = ("T", _attr_T, "align_corners", align_corners,
  "half_pixel_centers", half_pixel_centers)
  _result = _execute.execute(b"ResizeBilinear", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ResizeBilinear", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ResizeBilinearGrad_T = TypeVar("TV_ResizeBilinearGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def resize_bilinear_grad(grads: Annotated[Any, _atypes.Float32], original_image: Annotated[Any, TV_ResizeBilinearGrad_T], align_corners:bool=False, half_pixel_centers:bool=False, name=None) -> Annotated[Any, TV_ResizeBilinearGrad_T]:
  r"""Computes the gradient of bilinear interpolation.

  Args:
    grads: A `Tensor` of type `float32`.
      4-D with shape `[batch, height, width, channels]`.
    original_image: A `Tensor`. Must be one of the following types: `float32`, `bfloat16`, `half`, `float64`.
      4-D with shape `[batch, orig_height, orig_width, channels]`,
      The image tensor that was resized.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, the centers of the 4 corner pixels of the input and grad tensors are
      aligned. Defaults to false.
    half_pixel_centers: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `original_image`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResizeBilinearGrad", name, grads, original_image,
        "align_corners", align_corners, "half_pixel_centers",
        half_pixel_centers)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resize_bilinear_grad_eager_fallback(
          grads, original_image, align_corners=align_corners,
          half_pixel_centers=half_pixel_centers, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if align_corners is None:
    align_corners = False
  align_corners = _execute.make_bool(align_corners, "align_corners")
  if half_pixel_centers is None:
    half_pixel_centers = False
  half_pixel_centers = _execute.make_bool(half_pixel_centers, "half_pixel_centers")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResizeBilinearGrad", grads=grads, original_image=original_image,
                              align_corners=align_corners,
                              half_pixel_centers=half_pixel_centers,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "align_corners",
              _op._get_attr_bool("align_corners"), "half_pixel_centers",
              _op._get_attr_bool("half_pixel_centers"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ResizeBilinearGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ResizeBilinearGrad = tf_export("raw_ops.ResizeBilinearGrad")(_ops.to_raw_op(resize_bilinear_grad))


def resize_bilinear_grad_eager_fallback(grads: Annotated[Any, _atypes.Float32], original_image: Annotated[Any, TV_ResizeBilinearGrad_T], align_corners: bool, half_pixel_centers: bool, name, ctx) -> Annotated[Any, TV_ResizeBilinearGrad_T]:
  if align_corners is None:
    align_corners = False
  align_corners = _execute.make_bool(align_corners, "align_corners")
  if half_pixel_centers is None:
    half_pixel_centers = False
  half_pixel_centers = _execute.make_bool(half_pixel_centers, "half_pixel_centers")
  _attr_T, (original_image,) = _execute.args_to_matching_eager([original_image], ctx, [_dtypes.float32, _dtypes.bfloat16, _dtypes.half, _dtypes.float64, ])
  grads = _ops.convert_to_tensor(grads, _dtypes.float32)
  _inputs_flat = [grads, original_image]
  _attrs = ("T", _attr_T, "align_corners", align_corners,
  "half_pixel_centers", half_pixel_centers)
  _result = _execute.execute(b"ResizeBilinearGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ResizeBilinearGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ResizeNearestNeighbor_T = TypeVar("TV_ResizeNearestNeighbor_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt8)

def resize_nearest_neighbor(images: Annotated[Any, TV_ResizeNearestNeighbor_T], size: Annotated[Any, _atypes.Int32], align_corners:bool=False, half_pixel_centers:bool=False, name=None) -> Annotated[Any, TV_ResizeNearestNeighbor_T]:
  r"""Resize `images` to `size` using nearest neighbor interpolation.

  Args:
    images: A `Tensor`. Must be one of the following types: `int8`, `uint8`, `int16`, `uint16`, `int32`, `int64`, `half`, `float32`, `float64`, `bfloat16`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, the centers of the 4 corner pixels of the input and output tensors are
      aligned, preserving the values at the corner pixels. Defaults to false.
    half_pixel_centers: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResizeNearestNeighbor", name, images, size, "align_corners",
        align_corners, "half_pixel_centers", half_pixel_centers)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resize_nearest_neighbor_eager_fallback(
          images, size, align_corners=align_corners,
          half_pixel_centers=half_pixel_centers, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if align_corners is None:
    align_corners = False
  align_corners = _execute.make_bool(align_corners, "align_corners")
  if half_pixel_centers is None:
    half_pixel_centers = False
  half_pixel_centers = _execute.make_bool(half_pixel_centers, "half_pixel_centers")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResizeNearestNeighbor", images=images, size=size,
                                 align_corners=align_corners,
                                 half_pixel_centers=half_pixel_centers,
                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "align_corners",
              _op._get_attr_bool("align_corners"), "half_pixel_centers",
              _op._get_attr_bool("half_pixel_centers"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ResizeNearestNeighbor", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ResizeNearestNeighbor = tf_export("raw_ops.ResizeNearestNeighbor")(_ops.to_raw_op(resize_nearest_neighbor))


def resize_nearest_neighbor_eager_fallback(images: Annotated[Any, TV_ResizeNearestNeighbor_T], size: Annotated[Any, _atypes.Int32], align_corners: bool, half_pixel_centers: bool, name, ctx) -> Annotated[Any, TV_ResizeNearestNeighbor_T]:
  if align_corners is None:
    align_corners = False
  align_corners = _execute.make_bool(align_corners, "align_corners")
  if half_pixel_centers is None:
    half_pixel_centers = False
  half_pixel_centers = _execute.make_bool(half_pixel_centers, "half_pixel_centers")
  _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.int8, _dtypes.uint8, _dtypes.int16, _dtypes.uint16, _dtypes.int32, _dtypes.int64, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.bfloat16, ])
  size = _ops.convert_to_tensor(size, _dtypes.int32)
  _inputs_flat = [images, size]
  _attrs = ("T", _attr_T, "align_corners", align_corners,
  "half_pixel_centers", half_pixel_centers)
  _result = _execute.execute(b"ResizeNearestNeighbor", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ResizeNearestNeighbor", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ResizeNearestNeighborGrad_T = TypeVar("TV_ResizeNearestNeighborGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int8, _atypes.UInt8)

def resize_nearest_neighbor_grad(grads: Annotated[Any, TV_ResizeNearestNeighborGrad_T], size: Annotated[Any, _atypes.Int32], align_corners:bool=False, half_pixel_centers:bool=False, name=None) -> Annotated[Any, TV_ResizeNearestNeighborGrad_T]:
  r"""Computes the gradient of nearest neighbor interpolation.

  Args:
    grads: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `half`, `float32`, `float64`, `bfloat16`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `orig_height, orig_width`. The
      original input size.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, the centers of the 4 corner pixels of the input and grad tensors are
      aligned. Defaults to false.
    half_pixel_centers: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grads`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResizeNearestNeighborGrad", name, grads, size, "align_corners",
        align_corners, "half_pixel_centers", half_pixel_centers)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return resize_nearest_neighbor_grad_eager_fallback(
          grads, size, align_corners=align_corners,
          half_pixel_centers=half_pixel_centers, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if align_corners is None:
    align_corners = False
  align_corners = _execute.make_bool(align_corners, "align_corners")
  if half_pixel_centers is None:
    half_pixel_centers = False
  half_pixel_centers = _execute.make_bool(half_pixel_centers, "half_pixel_centers")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResizeNearestNeighborGrad", grads=grads, size=size,
                                     align_corners=align_corners,
                                     half_pixel_centers=half_pixel_centers,
                                     name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "align_corners",
              _op._get_attr_bool("align_corners"), "half_pixel_centers",
              _op._get_attr_bool("half_pixel_centers"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ResizeNearestNeighborGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ResizeNearestNeighborGrad = tf_export("raw_ops.ResizeNearestNeighborGrad")(_ops.to_raw_op(resize_nearest_neighbor_grad))


def resize_nearest_neighbor_grad_eager_fallback(grads: Annotated[Any, TV_ResizeNearestNeighborGrad_T], size: Annotated[Any, _atypes.Int32], align_corners: bool, half_pixel_centers: bool, name, ctx) -> Annotated[Any, TV_ResizeNearestNeighborGrad_T]:
  if align_corners is None:
    align_corners = False
  align_corners = _execute.make_bool(align_corners, "align_corners")
  if half_pixel_centers is None:
    half_pixel_centers = False
  half_pixel_centers = _execute.make_bool(half_pixel_centers, "half_pixel_centers")
  _attr_T, (grads,) = _execute.args_to_matching_eager([grads], ctx, [_dtypes.uint8, _dtypes.int8, _dtypes.int32, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.bfloat16, ])
  size = _ops.convert_to_tensor(size, _dtypes.int32)
  _inputs_flat = [grads, size]
  _attrs = ("T", _attr_T, "align_corners", align_corners,
  "half_pixel_centers", half_pixel_centers)
  _result = _execute.execute(b"ResizeNearestNeighborGrad", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ResizeNearestNeighborGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_SampleDistortedBoundingBoxOutput = collections.namedtuple(
    "SampleDistortedBoundingBox",
    ["begin", "size", "bboxes"])


TV_SampleDistortedBoundingBox_T = TypeVar("TV_SampleDistortedBoundingBox_T", _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt8)

def sample_distorted_bounding_box(image_size: Annotated[Any, TV_SampleDistortedBoundingBox_T], bounding_boxes: Annotated[Any, _atypes.Float32], seed:int=0, seed2:int=0, min_object_covered:float=0.1, aspect_ratio_range=[0.75, 1.33], area_range=[0.05, 1], max_attempts:int=100, use_image_if_no_bounding_boxes:bool=False, name=None):
  r"""Generate a single randomly distorted bounding box for an image.

  Bounding box annotations are often supplied in addition to ground-truth labels
  in image recognition or object localization tasks. A common technique for
  training such a system is to randomly distort an image while preserving
  its content, i.e. *data augmentation*. This Op outputs a randomly distorted
  localization of an object, i.e. bounding box, given an `image_size`,
  `bounding_boxes` and a series of constraints.

  The output of this Op is a single bounding box that may be used to crop the
  original image. The output is returned as 3 tensors: `begin`, `size` and
  `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
  image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
  what the bounding box looks like.

  Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
  bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
  height of the underlying image.

  For example,

  ```python
      # Generate a single distorted bounding box.
      begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
          tf.shape(image),
          bounding_boxes=bounding_boxes)

      # Draw the bounding box in an image summary.
      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                    bbox_for_draw)
      tf.summary.image('images_with_box', image_with_box)

      # Employ the bounding box to distort the image.
      distorted_image = tf.slice(image, begin, size)
  ```

  Note that if no bounding box information is available, setting
  `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
  bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
  false and no bounding boxes are supplied, an error is raised.

  Args:
    image_size: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`.
      1-D, containing `[height, width, channels]`.
    bounding_boxes: A `Tensor` of type `float32`.
      3-D with shape `[batch, N, 4]` describing the N bounding boxes
      associated with the image.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to non-zero, the random number
      generator is seeded by the given `seed`.  Otherwise, it is seeded by a random
      seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    min_object_covered: An optional `float`. Defaults to `0.1`.
      The cropped area of the image must contain at least this
      fraction of any bounding box supplied. The value of this parameter should be
      non-negative. In the case of 0, the cropped area does not need to overlap
      any of the bounding boxes supplied.
    aspect_ratio_range: An optional list of `floats`. Defaults to `[0.75, 1.33]`.
      The cropped area of the image must have an aspect ratio =
      width / height within this range.
    area_range: An optional list of `floats`. Defaults to `[0.05, 1]`.
      The cropped area of the image must contain a fraction of the
      supplied image within this range.
    max_attempts: An optional `int`. Defaults to `100`.
      Number of attempts at generating a cropped region of the image
      of the specified constraints. After `max_attempts` failures, return the entire
      image.
    use_image_if_no_bounding_boxes: An optional `bool`. Defaults to `False`.
      Controls behavior if no bounding boxes supplied.
      If true, assume an implicit bounding box covering the whole input. If false,
      raise an error.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (begin, size, bboxes).

    begin: A `Tensor`. Has the same type as `image_size`.
    size: A `Tensor`. Has the same type as `image_size`.
    bboxes: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SampleDistortedBoundingBox", name, image_size, bounding_boxes,
        "seed", seed, "seed2", seed2, "min_object_covered",
        min_object_covered, "aspect_ratio_range", aspect_ratio_range,
        "area_range", area_range, "max_attempts", max_attempts,
        "use_image_if_no_bounding_boxes", use_image_if_no_bounding_boxes)
      _result = _SampleDistortedBoundingBoxOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sample_distorted_bounding_box_eager_fallback(
          image_size, bounding_boxes, seed=seed, seed2=seed2,
          min_object_covered=min_object_covered,
          aspect_ratio_range=aspect_ratio_range, area_range=area_range,
          max_attempts=max_attempts,
          use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if min_object_covered is None:
    min_object_covered = 0.1
  min_object_covered = _execute.make_float(min_object_covered, "min_object_covered")
  if aspect_ratio_range is None:
    aspect_ratio_range = [0.75, 1.33]
  if not isinstance(aspect_ratio_range, (list, tuple)):
    raise TypeError(
        "Expected list for 'aspect_ratio_range' argument to "
        "'sample_distorted_bounding_box' Op, not %r." % aspect_ratio_range)
  aspect_ratio_range = [_execute.make_float(_f, "aspect_ratio_range") for _f in aspect_ratio_range]
  if area_range is None:
    area_range = [0.05, 1]
  if not isinstance(area_range, (list, tuple)):
    raise TypeError(
        "Expected list for 'area_range' argument to "
        "'sample_distorted_bounding_box' Op, not %r." % area_range)
  area_range = [_execute.make_float(_f, "area_range") for _f in area_range]
  if max_attempts is None:
    max_attempts = 100
  max_attempts = _execute.make_int(max_attempts, "max_attempts")
  if use_image_if_no_bounding_boxes is None:
    use_image_if_no_bounding_boxes = False
  use_image_if_no_bounding_boxes = _execute.make_bool(use_image_if_no_bounding_boxes, "use_image_if_no_bounding_boxes")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SampleDistortedBoundingBox", image_size=image_size,
                                      bounding_boxes=bounding_boxes,
                                      seed=seed, seed2=seed2,
                                      min_object_covered=min_object_covered,
                                      aspect_ratio_range=aspect_ratio_range,
                                      area_range=area_range,
                                      max_attempts=max_attempts,
                                      use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes,
                                      name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "seed", _op._get_attr_int("seed"),
              "seed2", _op._get_attr_int("seed2"), "min_object_covered",
              _op.get_attr("min_object_covered"), "aspect_ratio_range",
              _op.get_attr("aspect_ratio_range"), "area_range",
              _op.get_attr("area_range"), "max_attempts",
              _op._get_attr_int("max_attempts"),
              "use_image_if_no_bounding_boxes",
              _op._get_attr_bool("use_image_if_no_bounding_boxes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SampleDistortedBoundingBox", _inputs_flat, _attrs, _result)
  _result = _SampleDistortedBoundingBoxOutput._make(_result)
  return _result

SampleDistortedBoundingBox = tf_export("raw_ops.SampleDistortedBoundingBox")(_ops.to_raw_op(sample_distorted_bounding_box))


def sample_distorted_bounding_box_eager_fallback(image_size: Annotated[Any, TV_SampleDistortedBoundingBox_T], bounding_boxes: Annotated[Any, _atypes.Float32], seed: int, seed2: int, min_object_covered: float, aspect_ratio_range, area_range, max_attempts: int, use_image_if_no_bounding_boxes: bool, name, ctx):
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if min_object_covered is None:
    min_object_covered = 0.1
  min_object_covered = _execute.make_float(min_object_covered, "min_object_covered")
  if aspect_ratio_range is None:
    aspect_ratio_range = [0.75, 1.33]
  if not isinstance(aspect_ratio_range, (list, tuple)):
    raise TypeError(
        "Expected list for 'aspect_ratio_range' argument to "
        "'sample_distorted_bounding_box' Op, not %r." % aspect_ratio_range)
  aspect_ratio_range = [_execute.make_float(_f, "aspect_ratio_range") for _f in aspect_ratio_range]
  if area_range is None:
    area_range = [0.05, 1]
  if not isinstance(area_range, (list, tuple)):
    raise TypeError(
        "Expected list for 'area_range' argument to "
        "'sample_distorted_bounding_box' Op, not %r." % area_range)
  area_range = [_execute.make_float(_f, "area_range") for _f in area_range]
  if max_attempts is None:
    max_attempts = 100
  max_attempts = _execute.make_int(max_attempts, "max_attempts")
  if use_image_if_no_bounding_boxes is None:
    use_image_if_no_bounding_boxes = False
  use_image_if_no_bounding_boxes = _execute.make_bool(use_image_if_no_bounding_boxes, "use_image_if_no_bounding_boxes")
  _attr_T, (image_size,) = _execute.args_to_matching_eager([image_size], ctx, [_dtypes.uint8, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, ])
  bounding_boxes = _ops.convert_to_tensor(bounding_boxes, _dtypes.float32)
  _inputs_flat = [image_size, bounding_boxes]
  _attrs = ("T", _attr_T, "seed", seed, "seed2", seed2, "min_object_covered",
  min_object_covered, "aspect_ratio_range", aspect_ratio_range, "area_range",
  area_range, "max_attempts", max_attempts, "use_image_if_no_bounding_boxes",
  use_image_if_no_bounding_boxes)
  _result = _execute.execute(b"SampleDistortedBoundingBox", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SampleDistortedBoundingBox", _inputs_flat, _attrs, _result)
  _result = _SampleDistortedBoundingBoxOutput._make(_result)
  return _result

_SampleDistortedBoundingBoxV2Output = collections.namedtuple(
    "SampleDistortedBoundingBoxV2",
    ["begin", "size", "bboxes"])


TV_SampleDistortedBoundingBoxV2_T = TypeVar("TV_SampleDistortedBoundingBoxV2_T", _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt8)

def sample_distorted_bounding_box_v2(image_size: Annotated[Any, TV_SampleDistortedBoundingBoxV2_T], bounding_boxes: Annotated[Any, _atypes.Float32], min_object_covered: Annotated[Any, _atypes.Float32], seed:int=0, seed2:int=0, aspect_ratio_range=[0.75, 1.33], area_range=[0.05, 1], max_attempts:int=100, use_image_if_no_bounding_boxes:bool=False, name=None):
  r"""Generate a single randomly distorted bounding box for an image.

  Bounding box annotations are often supplied in addition to ground-truth labels
  in image recognition or object localization tasks. A common technique for
  training such a system is to randomly distort an image while preserving
  its content, i.e. *data augmentation*. This Op outputs a randomly distorted
  localization of an object, i.e. bounding box, given an `image_size`,
  `bounding_boxes` and a series of constraints.

  The output of this Op is a single bounding box that may be used to crop the
  original image. The output is returned as 3 tensors: `begin`, `size` and
  `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
  image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
  what the bounding box looks like.

  Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
  bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
  height of the underlying image.

  For example,

  ```python
      # Generate a single distorted bounding box.
      begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
          tf.shape(image),
          bounding_boxes=bounding_boxes)

      # Draw the bounding box in an image summary.
      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                    bbox_for_draw)
      tf.summary.image('images_with_box', image_with_box)

      # Employ the bounding box to distort the image.
      distorted_image = tf.slice(image, begin, size)
  ```

  Note that if no bounding box information is available, setting
  `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
  bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
  false and no bounding boxes are supplied, an error is raised.

  Args:
    image_size: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`.
      1-D, containing `[height, width, channels]`.
    bounding_boxes: A `Tensor` of type `float32`.
      3-D with shape `[batch, N, 4]` describing the N bounding boxes
      associated with the image.
    min_object_covered: A `Tensor` of type `float32`.
      The cropped area of the image must contain at least this
      fraction of any bounding box supplied. The value of this parameter should be
      non-negative. In the case of 0, the cropped area does not need to overlap
      any of the bounding boxes supplied.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to non-zero, the random number
      generator is seeded by the given `seed`.  Otherwise, it is seeded by a random
      seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    aspect_ratio_range: An optional list of `floats`. Defaults to `[0.75, 1.33]`.
      The cropped area of the image must have an aspect ratio =
      width / height within this range.
    area_range: An optional list of `floats`. Defaults to `[0.05, 1]`.
      The cropped area of the image must contain a fraction of the
      supplied image within this range.
    max_attempts: An optional `int`. Defaults to `100`.
      Number of attempts at generating a cropped region of the image
      of the specified constraints. After `max_attempts` failures, return the entire
      image.
    use_image_if_no_bounding_boxes: An optional `bool`. Defaults to `False`.
      Controls behavior if no bounding boxes supplied.
      If true, assume an implicit bounding box covering the whole input. If false,
      raise an error.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (begin, size, bboxes).

    begin: A `Tensor`. Has the same type as `image_size`.
    size: A `Tensor`. Has the same type as `image_size`.
    bboxes: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SampleDistortedBoundingBoxV2", name, image_size,
        bounding_boxes, min_object_covered, "seed", seed, "seed2", seed2,
        "aspect_ratio_range", aspect_ratio_range, "area_range", area_range,
        "max_attempts", max_attempts, "use_image_if_no_bounding_boxes",
        use_image_if_no_bounding_boxes)
      _result = _SampleDistortedBoundingBoxV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sample_distorted_bounding_box_v2_eager_fallback(
          image_size, bounding_boxes, min_object_covered, seed=seed,
          seed2=seed2, aspect_ratio_range=aspect_ratio_range,
          area_range=area_range, max_attempts=max_attempts,
          use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if aspect_ratio_range is None:
    aspect_ratio_range = [0.75, 1.33]
  if not isinstance(aspect_ratio_range, (list, tuple)):
    raise TypeError(
        "Expected list for 'aspect_ratio_range' argument to "
        "'sample_distorted_bounding_box_v2' Op, not %r." % aspect_ratio_range)
  aspect_ratio_range = [_execute.make_float(_f, "aspect_ratio_range") for _f in aspect_ratio_range]
  if area_range is None:
    area_range = [0.05, 1]
  if not isinstance(area_range, (list, tuple)):
    raise TypeError(
        "Expected list for 'area_range' argument to "
        "'sample_distorted_bounding_box_v2' Op, not %r." % area_range)
  area_range = [_execute.make_float(_f, "area_range") for _f in area_range]
  if max_attempts is None:
    max_attempts = 100
  max_attempts = _execute.make_int(max_attempts, "max_attempts")
  if use_image_if_no_bounding_boxes is None:
    use_image_if_no_bounding_boxes = False
  use_image_if_no_bounding_boxes = _execute.make_bool(use_image_if_no_bounding_boxes, "use_image_if_no_bounding_boxes")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SampleDistortedBoundingBoxV2", image_size=image_size,
                                        bounding_boxes=bounding_boxes,
                                        min_object_covered=min_object_covered,
                                        seed=seed, seed2=seed2,
                                        aspect_ratio_range=aspect_ratio_range,
                                        area_range=area_range,
                                        max_attempts=max_attempts,
                                        use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes,
                                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "seed", _op._get_attr_int("seed"),
              "seed2", _op._get_attr_int("seed2"), "aspect_ratio_range",
              _op.get_attr("aspect_ratio_range"), "area_range",
              _op.get_attr("area_range"), "max_attempts",
              _op._get_attr_int("max_attempts"),
              "use_image_if_no_bounding_boxes",
              _op._get_attr_bool("use_image_if_no_bounding_boxes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SampleDistortedBoundingBoxV2", _inputs_flat, _attrs, _result)
  _result = _SampleDistortedBoundingBoxV2Output._make(_result)
  return _result

SampleDistortedBoundingBoxV2 = tf_export("raw_ops.SampleDistortedBoundingBoxV2")(_ops.to_raw_op(sample_distorted_bounding_box_v2))


def sample_distorted_bounding_box_v2_eager_fallback(image_size: Annotated[Any, TV_SampleDistortedBoundingBoxV2_T], bounding_boxes: Annotated[Any, _atypes.Float32], min_object_covered: Annotated[Any, _atypes.Float32], seed: int, seed2: int, aspect_ratio_range, area_range, max_attempts: int, use_image_if_no_bounding_boxes: bool, name, ctx):
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  if aspect_ratio_range is None:
    aspect_ratio_range = [0.75, 1.33]
  if not isinstance(aspect_ratio_range, (list, tuple)):
    raise TypeError(
        "Expected list for 'aspect_ratio_range' argument to "
        "'sample_distorted_bounding_box_v2' Op, not %r." % aspect_ratio_range)
  aspect_ratio_range = [_execute.make_float(_f, "aspect_ratio_range") for _f in aspect_ratio_range]
  if area_range is None:
    area_range = [0.05, 1]
  if not isinstance(area_range, (list, tuple)):
    raise TypeError(
        "Expected list for 'area_range' argument to "
        "'sample_distorted_bounding_box_v2' Op, not %r." % area_range)
  area_range = [_execute.make_float(_f, "area_range") for _f in area_range]
  if max_attempts is None:
    max_attempts = 100
  max_attempts = _execute.make_int(max_attempts, "max_attempts")
  if use_image_if_no_bounding_boxes is None:
    use_image_if_no_bounding_boxes = False
  use_image_if_no_bounding_boxes = _execute.make_bool(use_image_if_no_bounding_boxes, "use_image_if_no_bounding_boxes")
  _attr_T, (image_size,) = _execute.args_to_matching_eager([image_size], ctx, [_dtypes.uint8, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, ])
  bounding_boxes = _ops.convert_to_tensor(bounding_boxes, _dtypes.float32)
  min_object_covered = _ops.convert_to_tensor(min_object_covered, _dtypes.float32)
  _inputs_flat = [image_size, bounding_boxes, min_object_covered]
  _attrs = ("T", _attr_T, "seed", seed, "seed2", seed2, "aspect_ratio_range",
  aspect_ratio_range, "area_range", area_range, "max_attempts", max_attempts,
  "use_image_if_no_bounding_boxes", use_image_if_no_bounding_boxes)
  _result = _execute.execute(b"SampleDistortedBoundingBoxV2", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SampleDistortedBoundingBoxV2", _inputs_flat, _attrs, _result)
  _result = _SampleDistortedBoundingBoxV2Output._make(_result)
  return _result


TV_ScaleAndTranslate_T = TypeVar("TV_ScaleAndTranslate_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt8)

def scale_and_translate(images: Annotated[Any, TV_ScaleAndTranslate_T], size: Annotated[Any, _atypes.Int32], scale: Annotated[Any, _atypes.Float32], translation: Annotated[Any, _atypes.Float32], kernel_type:str="lanczos3", antialias:bool=True, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""TODO: add doc.

  Args:
    images: A `Tensor`. Must be one of the following types: `int8`, `uint8`, `int16`, `uint16`, `int32`, `int64`, `bfloat16`, `half`, `float32`, `float64`.
    size: A `Tensor` of type `int32`.
    scale: A `Tensor` of type `float32`.
    translation: A `Tensor` of type `float32`.
    kernel_type: An optional `string`. Defaults to `"lanczos3"`.
    antialias: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ScaleAndTranslate", name, images, size, scale, translation,
        "kernel_type", kernel_type, "antialias", antialias)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return scale_and_translate_eager_fallback(
          images, size, scale, translation, kernel_type=kernel_type,
          antialias=antialias, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if kernel_type is None:
    kernel_type = "lanczos3"
  kernel_type = _execute.make_str(kernel_type, "kernel_type")
  if antialias is None:
    antialias = True
  antialias = _execute.make_bool(antialias, "antialias")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ScaleAndTranslate", images=images, size=size, scale=scale,
                             translation=translation, kernel_type=kernel_type,
                             antialias=antialias, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "kernel_type",
              _op.get_attr("kernel_type"), "antialias",
              _op._get_attr_bool("antialias"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ScaleAndTranslate", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ScaleAndTranslate = tf_export("raw_ops.ScaleAndTranslate")(_ops.to_raw_op(scale_and_translate))


def scale_and_translate_eager_fallback(images: Annotated[Any, TV_ScaleAndTranslate_T], size: Annotated[Any, _atypes.Int32], scale: Annotated[Any, _atypes.Float32], translation: Annotated[Any, _atypes.Float32], kernel_type: str, antialias: bool, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if kernel_type is None:
    kernel_type = "lanczos3"
  kernel_type = _execute.make_str(kernel_type, "kernel_type")
  if antialias is None:
    antialias = True
  antialias = _execute.make_bool(antialias, "antialias")
  _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.int8, _dtypes.uint8, _dtypes.int16, _dtypes.uint16, _dtypes.int32, _dtypes.int64, _dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  size = _ops.convert_to_tensor(size, _dtypes.int32)
  scale = _ops.convert_to_tensor(scale, _dtypes.float32)
  translation = _ops.convert_to_tensor(translation, _dtypes.float32)
  _inputs_flat = [images, size, scale, translation]
  _attrs = ("T", _attr_T, "kernel_type", kernel_type, "antialias", antialias)
  _result = _execute.execute(b"ScaleAndTranslate", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ScaleAndTranslate", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ScaleAndTranslateGrad_T = TypeVar("TV_ScaleAndTranslateGrad_T", bound=_atypes.Float32)

def scale_and_translate_grad(grads: Annotated[Any, TV_ScaleAndTranslateGrad_T], original_image: Annotated[Any, TV_ScaleAndTranslateGrad_T], scale: Annotated[Any, _atypes.Float32], translation: Annotated[Any, _atypes.Float32], kernel_type:str="lanczos3", antialias:bool=True, name=None) -> Annotated[Any, TV_ScaleAndTranslateGrad_T]:
  r"""TODO: add doc.

  Args:
    grads: A `Tensor`. Must be one of the following types: `float32`.
    original_image: A `Tensor`. Must have the same type as `grads`.
    scale: A `Tensor` of type `float32`.
    translation: A `Tensor` of type `float32`.
    kernel_type: An optional `string`. Defaults to `"lanczos3"`.
    antialias: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grads`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ScaleAndTranslateGrad", name, grads, original_image, scale,
        translation, "kernel_type", kernel_type, "antialias", antialias)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return scale_and_translate_grad_eager_fallback(
          grads, original_image, scale, translation, kernel_type=kernel_type,
          antialias=antialias, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if kernel_type is None:
    kernel_type = "lanczos3"
  kernel_type = _execute.make_str(kernel_type, "kernel_type")
  if antialias is None:
    antialias = True
  antialias = _execute.make_bool(antialias, "antialias")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ScaleAndTranslateGrad", grads=grads, original_image=original_image,
                                 scale=scale, translation=translation,
                                 kernel_type=kernel_type, antialias=antialias,
                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "kernel_type",
              _op.get_attr("kernel_type"), "antialias",
              _op._get_attr_bool("antialias"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ScaleAndTranslateGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ScaleAndTranslateGrad = tf_export("raw_ops.ScaleAndTranslateGrad")(_ops.to_raw_op(scale_and_translate_grad))


def scale_and_translate_grad_eager_fallback(grads: Annotated[Any, TV_ScaleAndTranslateGrad_T], original_image: Annotated[Any, TV_ScaleAndTranslateGrad_T], scale: Annotated[Any, _atypes.Float32], translation: Annotated[Any, _atypes.Float32], kernel_type: str, antialias: bool, name, ctx) -> Annotated[Any, TV_ScaleAndTranslateGrad_T]:
  if kernel_type is None:
    kernel_type = "lanczos3"
  kernel_type = _execute.make_str(kernel_type, "kernel_type")
  if antialias is None:
    antialias = True
  antialias = _execute.make_bool(antialias, "antialias")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([grads, original_image], ctx, [_dtypes.float32, ])
  (grads, original_image) = _inputs_T
  scale = _ops.convert_to_tensor(scale, _dtypes.float32)
  translation = _ops.convert_to_tensor(translation, _dtypes.float32)
  _inputs_flat = [grads, original_image, scale, translation]
  _attrs = ("T", _attr_T, "kernel_type", kernel_type, "antialias", antialias)
  _result = _execute.execute(b"ScaleAndTranslateGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ScaleAndTranslateGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_StatelessSampleDistortedBoundingBoxOutput = collections.namedtuple(
    "StatelessSampleDistortedBoundingBox",
    ["begin", "size", "bboxes"])


TV_StatelessSampleDistortedBoundingBox_T = TypeVar("TV_StatelessSampleDistortedBoundingBox_T", _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt8)
TV_StatelessSampleDistortedBoundingBox_Tseed = TypeVar("TV_StatelessSampleDistortedBoundingBox_Tseed", _atypes.Int32, _atypes.Int64)

def stateless_sample_distorted_bounding_box(image_size: Annotated[Any, TV_StatelessSampleDistortedBoundingBox_T], bounding_boxes: Annotated[Any, _atypes.Float32], min_object_covered: Annotated[Any, _atypes.Float32], seed: Annotated[Any, TV_StatelessSampleDistortedBoundingBox_Tseed], aspect_ratio_range=[0.75, 1.33], area_range=[0.05, 1], max_attempts:int=100, use_image_if_no_bounding_boxes:bool=False, name=None):
  r"""Generate a randomly distorted bounding box for an image deterministically.

  Bounding box annotations are often supplied in addition to ground-truth labels
  in image recognition or object localization tasks. A common technique for
  training such a system is to randomly distort an image while preserving its
  content, i.e. *data augmentation*. This Op, given the same `seed`,
  deterministically outputs a randomly distorted localization of an object, i.e.
  bounding box, given an `image_size`, `bounding_boxes` and a series of
  constraints.

  The output of this Op is a single bounding box that may be used to crop the
  original image. The output is returned as 3 tensors: `begin`, `size` and
  `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
  image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
  what the bounding box looks like.

  Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
  bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
  the height of the underlying image.

  The output of this Op is guaranteed to be the same given the same `seed` and is
  independent of how many times the function is called, and independent of global
  seed settings (e.g. `tf.random.set_seed`).

  Example usage:

  >>> image = np.array([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])
  >>> bbox = tf.constant(
  ...   [0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  >>> seed = (1, 2)
  >>> # Generate a single distorted bounding box.
  >>> bbox_begin, bbox_size, bbox_draw = (
  ...   tf.image.stateless_sample_distorted_bounding_box(
  ...     tf.shape(image), bounding_boxes=bbox, seed=seed))
  >>> # Employ the bounding box to distort the image.
  >>> tf.slice(image, bbox_begin, bbox_size)
  <tf.Tensor: shape=(2, 2, 1), dtype=int64, numpy=
  array([[[1],
          [2]],
         [[4],
          [5]]])>
  >>> # Draw the bounding box in an image summary.
  >>> colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
  >>> tf.image.draw_bounding_boxes(
  ...   tf.expand_dims(tf.cast(image, tf.float32),0), bbox_draw, colors)
  <tf.Tensor: shape=(1, 3, 3, 1), dtype=float32, numpy=
  array([[[[1.],
           [1.],
           [3.]],
          [[1.],
           [1.],
           [6.]],
          [[7.],
           [8.],
           [9.]]]], dtype=float32)>

  Note that if no bounding box information is available, setting
  `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
  bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
  false and no bounding boxes are supplied, an error is raised.

  Args:
    image_size: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`.
      1-D, containing `[height, width, channels]`.
    bounding_boxes: A `Tensor` of type `float32`.
      3-D with shape `[batch, N, 4]` describing the N bounding boxes
      associated with the image.
    min_object_covered: A `Tensor` of type `float32`.
      The cropped area of the image must contain at least this
      fraction of any bounding box supplied. The value of this parameter should be
      non-negative. In the case of 0, the cropped area does not need to overlap
      any of the bounding boxes supplied.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1-D with shape `[2]`. The seed to the random number generator. Must have dtype
      `int32` or `int64`. (When using XLA, only `int32` is allowed.)
    aspect_ratio_range: An optional list of `floats`. Defaults to `[0.75, 1.33]`.
      The cropped area of the image must have an aspect ratio =
      width / height within this range.
    area_range: An optional list of `floats`. Defaults to `[0.05, 1]`.
      The cropped area of the image must contain a fraction of the
      supplied image within this range.
    max_attempts: An optional `int`. Defaults to `100`.
      Number of attempts at generating a cropped region of the image
      of the specified constraints. After `max_attempts` failures, return the entire
      image.
    use_image_if_no_bounding_boxes: An optional `bool`. Defaults to `False`.
      Controls behavior if no bounding boxes supplied.
      If true, assume an implicit bounding box covering the whole input. If false,
      raise an error.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (begin, size, bboxes).

    begin: A `Tensor`. Has the same type as `image_size`.
    size: A `Tensor`. Has the same type as `image_size`.
    bboxes: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StatelessSampleDistortedBoundingBox", name, image_size,
        bounding_boxes, min_object_covered, seed, "aspect_ratio_range",
        aspect_ratio_range, "area_range", area_range, "max_attempts",
        max_attempts, "use_image_if_no_bounding_boxes",
        use_image_if_no_bounding_boxes)
      _result = _StatelessSampleDistortedBoundingBoxOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return stateless_sample_distorted_bounding_box_eager_fallback(
          image_size, bounding_boxes, min_object_covered, seed,
          aspect_ratio_range=aspect_ratio_range, area_range=area_range,
          max_attempts=max_attempts,
          use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if aspect_ratio_range is None:
    aspect_ratio_range = [0.75, 1.33]
  if not isinstance(aspect_ratio_range, (list, tuple)):
    raise TypeError(
        "Expected list for 'aspect_ratio_range' argument to "
        "'stateless_sample_distorted_bounding_box' Op, not %r." % aspect_ratio_range)
  aspect_ratio_range = [_execute.make_float(_f, "aspect_ratio_range") for _f in aspect_ratio_range]
  if area_range is None:
    area_range = [0.05, 1]
  if not isinstance(area_range, (list, tuple)):
    raise TypeError(
        "Expected list for 'area_range' argument to "
        "'stateless_sample_distorted_bounding_box' Op, not %r." % area_range)
  area_range = [_execute.make_float(_f, "area_range") for _f in area_range]
  if max_attempts is None:
    max_attempts = 100
  max_attempts = _execute.make_int(max_attempts, "max_attempts")
  if use_image_if_no_bounding_boxes is None:
    use_image_if_no_bounding_boxes = False
  use_image_if_no_bounding_boxes = _execute.make_bool(use_image_if_no_bounding_boxes, "use_image_if_no_bounding_boxes")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StatelessSampleDistortedBoundingBox", image_size=image_size,
                                               bounding_boxes=bounding_boxes,
                                               min_object_covered=min_object_covered,
                                               seed=seed,
                                               aspect_ratio_range=aspect_ratio_range,
                                               area_range=area_range,
                                               max_attempts=max_attempts,
                                               use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes,
                                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tseed",
              _op._get_attr_type("Tseed"), "aspect_ratio_range",
              _op.get_attr("aspect_ratio_range"), "area_range",
              _op.get_attr("area_range"), "max_attempts",
              _op._get_attr_int("max_attempts"),
              "use_image_if_no_bounding_boxes",
              _op._get_attr_bool("use_image_if_no_bounding_boxes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StatelessSampleDistortedBoundingBox", _inputs_flat, _attrs, _result)
  _result = _StatelessSampleDistortedBoundingBoxOutput._make(_result)
  return _result

StatelessSampleDistortedBoundingBox = tf_export("raw_ops.StatelessSampleDistortedBoundingBox")(_ops.to_raw_op(stateless_sample_distorted_bounding_box))


def stateless_sample_distorted_bounding_box_eager_fallback(image_size: Annotated[Any, TV_StatelessSampleDistortedBoundingBox_T], bounding_boxes: Annotated[Any, _atypes.Float32], min_object_covered: Annotated[Any, _atypes.Float32], seed: Annotated[Any, TV_StatelessSampleDistortedBoundingBox_Tseed], aspect_ratio_range, area_range, max_attempts: int, use_image_if_no_bounding_boxes: bool, name, ctx):
  if aspect_ratio_range is None:
    aspect_ratio_range = [0.75, 1.33]
  if not isinstance(aspect_ratio_range, (list, tuple)):
    raise TypeError(
        "Expected list for 'aspect_ratio_range' argument to "
        "'stateless_sample_distorted_bounding_box' Op, not %r." % aspect_ratio_range)
  aspect_ratio_range = [_execute.make_float(_f, "aspect_ratio_range") for _f in aspect_ratio_range]
  if area_range is None:
    area_range = [0.05, 1]
  if not isinstance(area_range, (list, tuple)):
    raise TypeError(
        "Expected list for 'area_range' argument to "
        "'stateless_sample_distorted_bounding_box' Op, not %r." % area_range)
  area_range = [_execute.make_float(_f, "area_range") for _f in area_range]
  if max_attempts is None:
    max_attempts = 100
  max_attempts = _execute.make_int(max_attempts, "max_attempts")
  if use_image_if_no_bounding_boxes is None:
    use_image_if_no_bounding_boxes = False
  use_image_if_no_bounding_boxes = _execute.make_bool(use_image_if_no_bounding_boxes, "use_image_if_no_bounding_boxes")
  _attr_T, (image_size,) = _execute.args_to_matching_eager([image_size], ctx, [_dtypes.uint8, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, ])
  _attr_Tseed, (seed,) = _execute.args_to_matching_eager([seed], ctx, [_dtypes.int32, _dtypes.int64, ])
  bounding_boxes = _ops.convert_to_tensor(bounding_boxes, _dtypes.float32)
  min_object_covered = _ops.convert_to_tensor(min_object_covered, _dtypes.float32)
  _inputs_flat = [image_size, bounding_boxes, min_object_covered, seed]
  _attrs = ("T", _attr_T, "Tseed", _attr_Tseed, "aspect_ratio_range",
  aspect_ratio_range, "area_range", area_range, "max_attempts", max_attempts,
  "use_image_if_no_bounding_boxes", use_image_if_no_bounding_boxes)
  _result = _execute.execute(b"StatelessSampleDistortedBoundingBox", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StatelessSampleDistortedBoundingBox", _inputs_flat, _attrs, _result)
  _result = _StatelessSampleDistortedBoundingBoxOutput._make(_result)
  return _result

