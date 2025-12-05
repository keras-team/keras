# Copyright 2024 The Keras Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.losses.loss import Loss
from keras.src.losses.losses import LossFunctionWrapper


def _build_vgg16_feature_extractor(layer_names=None, weights=None):
    # Lazy import to avoid heavy dependencies during package import
    from keras.src.applications.vgg16 import VGG16
    from keras.src import models

    if layer_names is None:
        # Standard LPIPS uses conv2 from blocks 1,2 and conv3 from blocks 3,4,5
        layer_names = [
            "block1_conv2",
            "block2_conv2",
            "block3_conv3",
            "block4_conv3",
            "block5_conv3",
        ]
    base = VGG16(include_top=False, weights=weights)
    outputs = [base.get_layer(name).output for name in layer_names]
    # Create a model that returns a list of intermediate activations
    feat_model = models.Model(inputs=base.input, outputs=outputs, name="vgg16_lpips")
    feat_model.trainable = False
    return feat_model


def _normalize_channels(x, epsilon=1e-6):
    # Per-channel L2 normalization across spatial dimensions H, W
    # x: (B, H, W, C)
    # Compute norm over H,W for each channel
    hw_axes = tuple(range(1, x.ndim - 1))
    norm = ops.sqrt(ops.sum(ops.square(x), axis=hw_axes, keepdims=True) + epsilon)
    return x / norm


@keras_export("keras.losses.lpips")
def lpips(
    y_true,
    y_pred,
    feature_model=None,
    layer_weights=None,
    normalize_input=True,
):
    """Computes a perceptual distance between images using feature activations.

    This is an approximation of LPIPS using a fixed feature extractor
    (default: VGG16 conv blocks). It avoids network access by default by not
    loading any pretrained weights unless a `feature_model` with weights is
    provided by the user.

    Args:
        y_true: Tensor of reference images, shape (batch, H, W, 3), values in
            [0, 1] or [-1, 1].
        y_pred: Tensor of compared images, same shape and dtype as `y_true`.
        feature_model: Optional Keras model that maps an image tensor to a
            list/tuple of feature maps. If None, a VGG16-based extractor is
            constructed internally with `weights=None`.
        layer_weights: Optional list of scalars for each feature map. If None,
            equal weights are used.
        normalize_input: If True, rescale inputs from [0, 1] to [-1, 1]. If the
            inputs already lie in [-1, 1], this is a no-op.

    Returns:
        A 1D tensor with one scalar perceptual distance per sample.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)

    # Ensure channel-last images
    if y_pred.ndim != 4 or y_pred.shape[-1] != 3:
        raise ValueError(
            "lpips expects inputs of shape (batch, H, W, 3) with channels-last."
        )

    # Normalize to [-1, 1] if requested and inputs appear to be in [0,1]
    if normalize_input:
        # Heuristic: if max value <= 1.5, assume [0,1] and map to [-1,1]
        # Use ops to be backend-agnostic
        max_val = ops.max(ops.maximum(y_true, y_pred))
        cond = ops.less_equal(max_val, ops.convert_to_tensor(1.5, y_pred.dtype))

        def _scale_to_m1_1(x):
            return x * 2.0 - 1.0

        y_true = ops.cond(cond, lambda: _scale_to_m1_1(y_true), lambda: y_true)
        y_pred = ops.cond(cond, lambda: _scale_to_m1_1(y_pred), lambda: y_pred)

    # Build default feature extractor if not provided
    if feature_model is None:
        feature_model = _build_vgg16_feature_extractor(weights=None)

    # Resize inputs to the model input size if necessary
    target_h, target_w = feature_model.input_shape[1], feature_model.input_shape[2]
    if (target_h is not None and target_w is not None) and (
        y_true.shape[1] != target_h or y_true.shape[2] != target_w
    ):
        from keras.src import layers

        y_true = layers.Resizing(int(target_h), int(target_w), interpolation="bilinear")(y_true)
        y_pred = layers.Resizing(int(target_h), int(target_w), interpolation="bilinear")(y_pred)

    # Forward pass to get feature lists
    feats_true = feature_model(y_true)
    feats_pred = feature_model(y_pred)

    # Ensure iterable
    if not isinstance(feats_true, (list, tuple)):
        feats_true = (feats_true,)
        feats_pred = (feats_pred,)

    if layer_weights is None:
        layer_weights = [1.0] * len(feats_true)
    else:
        if len(layer_weights) != len(feats_true):
            raise ValueError(
                "layer_weights length must match the number of feature maps"
            )

    # Compute per-layer distances and sum
    distances = []
    for w, f_t, f_p in zip(layer_weights, feats_true, feats_pred):
        f_t = ops.convert_to_tensor(f_t, dtype=y_pred.dtype)
        f_p = ops.convert_to_tensor(f_p, dtype=y_pred.dtype)
        # Channel-wise normalization
        f_t = _normalize_channels(f_t)
        f_p = _normalize_channels(f_p)
        diff = ops.square(f_t - f_p)
        # Average across spatial and channel dims -> per-sample scalar
        axes = tuple(range(1, diff.ndim))
        d = ops.mean(diff, axis=axes)
        distances.append(w * d)

    total = distances[0]
    for d in distances[1:]:
        total = total + d
    return total


@keras_export("keras.losses.LPIPS")
class LPIPS(LossFunctionWrapper):
    """Perceptual distance loss using deep feature activations.

    This provides a backend-agnostic approximation of the LPIPS loss.
    By default it uses a VGG16-based feature extractor with random weights
    (no downloads) to keep tests and offline usage lightweight. For more
    accurate behavior, pass in a pretrained `feature_model` and optional
    `layer_weights`.

    Args:
        feature_model: Optional Keras model mapping an image to a list of
            feature maps. If None, a VGG16-based extractor is constructed with
            `weights=None`.
        layer_weights: Optional list of scalars to weight each feature map.
        normalize_input: Whether to map inputs from [0,1] to [-1,1].
        reduction: Loss reduction. Defaults to "sum_over_batch_size".
        name: Optional name for this loss.
        dtype: Dtype for computations.
    """

    def __init__(
        self,
        feature_model=None,
        layer_weights=None,
        normalize_input=True,
        reduction="sum_over_batch_size",
        name="lpips",
        dtype=None,
    ):
        super().__init__(
            lpips,
            name=name,
            reduction=reduction,
            dtype=dtype,
            feature_model=feature_model,
            layer_weights=layer_weights,
            normalize_input=normalize_input,
        )
        self._has_custom_model = feature_model is not None
        self.layer_weights = layer_weights
        self.normalize_input = normalize_input

    def get_config(self):
        # We cannot reliably serialize a custom feature_model; only config
        # for behavior flags is returned.
        config = Loss.get_config(self)
        config.update(
            {
                "feature_model": None if self._has_custom_model else "vgg16",
                "layer_weights": self.layer_weights,
                "normalize_input": self.normalize_input,
            }
        )
        return config