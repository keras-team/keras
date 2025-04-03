# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Image summaries and TensorFlow operations to create them.

An image summary stores the width, height, and PNG-encoded data for zero
or more images in a rank-1 string array: `[w, h, png0, png1, ...]`.

NOTE: This module is in beta, and its API is subject to change, but the
data that it stores to disk will be supported forever.
"""


import numpy as np

from tensorboard.plugins.image import metadata
from tensorboard.plugins.image import summary_v2
from tensorboard.util import encoder


# Export V2 versions.
image = summary_v2.image


def op(
    name,
    images,
    max_outputs=3,
    display_name=None,
    description=None,
    collections=None,
):
    """Create a legacy image summary op for use in a TensorFlow graph.

    Arguments:
      name: A unique name for the generated summary node.
      images: A `Tensor` representing pixel data with shape `[k, h, w, c]`,
        where `k` is the number of images, `h` and `w` are the height and
        width of the images, and `c` is the number of channels, which
        should be 1, 3, or 4. Any of the dimensions may be statically
        unknown (i.e., `None`).
      max_outputs: Optional `int` or rank-0 integer `Tensor`. At most this
        many images will be emitted at each step. When more than
        `max_outputs` many images are provided, the first `max_outputs` many
        images will be used and the rest silently discarded.
      display_name: Optional name for this summary in TensorBoard, as a
        constant `str`. Defaults to `name`.
      description: Optional long-form description for this summary, as a
        constant `str`. Markdown is supported. Defaults to empty.
      collections: Optional list of graph collections keys. The new
        summary op is added to these collections. Defaults to
        `[Graph Keys.SUMMARIES]`.

    Returns:
      A TensorFlow summary op.
    """
    # TODO(nickfelt): remove on-demand imports once dep situation is fixed.
    import tensorflow.compat.v1 as tf

    if display_name is None:
        display_name = name
    summary_metadata = metadata.create_summary_metadata(
        display_name=display_name, description=description
    )
    with tf.name_scope(name), tf.control_dependencies(
        [
            tf.assert_rank(images, 4),
            tf.assert_type(images, tf.uint8),
            tf.assert_non_negative(max_outputs),
        ]
    ):
        limited_images = images[:max_outputs]
        encoded_images = tf.map_fn(
            tf.image.encode_png,
            limited_images,
            dtype=tf.string,
            name="encode_each_image",
        )
        image_shape = tf.shape(input=images)
        dimensions = tf.stack(
            [
                tf.as_string(image_shape[2], name="width"),
                tf.as_string(image_shape[1], name="height"),
            ],
            name="dimensions",
        )
        tensor = tf.concat([dimensions, encoded_images], axis=0)
        return tf.summary.tensor_summary(
            name="image_summary",
            tensor=tensor,
            collections=collections,
            summary_metadata=summary_metadata,
        )


def pb(name, images, max_outputs=3, display_name=None, description=None):
    """Create a legacy image summary protobuf.

    This behaves as if you were to create an `op` with the same arguments
    (wrapped with constant tensors where appropriate) and then execute
    that summary op in a TensorFlow session.

    Arguments:
      name: A unique name for the generated summary, including any desired
        name scopes.
      images: An `np.array` representing pixel data with shape
        `[k, h, w, c]`, where `k` is the number of images, `w` and `h` are
        the width and height of the images, and `c` is the number of
        channels, which should be 1, 3, or 4.
      max_outputs: Optional `int`. At most this many images will be
        emitted. If more than this many images are provided, the first
        `max_outputs` many images will be used and the rest silently
        discarded.
      display_name: Optional name for this summary in TensorBoard, as a
        `str`. Defaults to `name`.
      description: Optional long-form description for this summary, as a
        `str`. Markdown is supported. Defaults to empty.

    Returns:
      A `tf.Summary` protobuf object.
    """
    # TODO(nickfelt): remove on-demand imports once dep situation is fixed.
    import tensorflow.compat.v1 as tf

    images = np.array(images).astype(np.uint8)
    if images.ndim != 4:
        raise ValueError("Shape %r must have rank 4" % (images.shape,))

    limited_images = images[:max_outputs]
    encoded_images = [encoder.encode_png(image) for image in limited_images]
    (width, height) = (images.shape[2], images.shape[1])
    content = [str(width), str(height)] + encoded_images
    tensor = tf.make_tensor_proto(content, dtype=tf.string)

    if display_name is None:
        display_name = name
    summary_metadata = metadata.create_summary_metadata(
        display_name=display_name, description=description
    )
    tf_summary_metadata = tf.SummaryMetadata.FromString(
        summary_metadata.SerializeToString()
    )

    summary = tf.Summary()
    summary.value.add(
        tag="%s/image_summary" % name,
        metadata=tf_summary_metadata,
        tensor=tensor,
    )
    return summary
