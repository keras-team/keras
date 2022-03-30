# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities related to image handling."""
# pylint: disable=g-direct-tensorflow-import
# pylint: disable=g-import-not-at-top

import io
import pathlib
import warnings

from keras import backend
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export

try:
  from PIL import Image as pil_image
except ImportError:
  pil_image = None


if pil_image is not None:
  _PIL_INTERPOLATION_METHODS = {
      'nearest': pil_image.NEAREST,
      'bilinear': pil_image.BILINEAR,
      'bicubic': pil_image.BICUBIC,
      'hamming': pil_image.HAMMING,
      'box': pil_image.BOX,
      'lanczos': pil_image.LANCZOS,
  }

ResizeMethod = tf.image.ResizeMethod

_TF_INTERPOLATION_METHODS = {
    'bilinear': ResizeMethod.BILINEAR,
    'nearest': ResizeMethod.NEAREST_NEIGHBOR,
    'bicubic': ResizeMethod.BICUBIC,
    'area': ResizeMethod.AREA,
    'lanczos3': ResizeMethod.LANCZOS3,
    'lanczos5': ResizeMethod.LANCZOS5,
    'gaussian': ResizeMethod.GAUSSIAN,
    'mitchellcubic': ResizeMethod.MITCHELLCUBIC
}


@keras_export('keras.preprocessing.image.smart_resize', v1=[])
def smart_resize(x, size, interpolation='bilinear'):
  """Resize images to a target size without aspect ratio distortion.

  Warning: `tf.keras.preprocessing.image.smart_resize` is not recommended for
  new code. Prefer `tf.keras.layers.Resizing`, which provides the same
  functionality as a preprocessing layer and adds `tf.RaggedTensor` support. See
  the [preprocessing layer guide](
  https://www.tensorflow.org/guide/keras/preprocessing_layers)
  for an overview of preprocessing layers.

  TensorFlow image datasets typically yield images that have each a different
  size. However, these images need to be batched before they can be
  processed by Keras layers. To be batched, images need to share the same height
  and width.

  You could simply do:

  ```python
  size = (200, 200)
  ds = ds.map(lambda img: tf.image.resize(img, size))
  ```

  However, if you do this, you distort the aspect ratio of your images, since
  in general they do not all have the same aspect ratio as `size`. This is
  fine in many cases, but not always (e.g. for GANs this can be a problem).

  Note that passing the argument `preserve_aspect_ratio=True` to `resize`
  will preserve the aspect ratio, but at the cost of no longer respecting the
  provided target size. Because `tf.image.resize` doesn't crop images,
  your output images will still have different sizes.

  This calls for:

  ```python
  size = (200, 200)
  ds = ds.map(lambda img: smart_resize(img, size))
  ```

  Your output images will actually be `(200, 200)`, and will not be distorted.
  Instead, the parts of the image that do not fit within the target size
  get cropped out.

  The resizing process is:

  1. Take the largest centered crop of the image that has the same aspect ratio
  as the target size. For instance, if `size=(200, 200)` and the input image has
  size `(340, 500)`, we take a crop of `(340, 340)` centered along the width.
  2. Resize the cropped image to the target size. In the example above,
  we resize the `(340, 340)` crop to `(200, 200)`.

  Args:
    x: Input image or batch of images (as a tensor or NumPy array). Must be in
      format `(height, width, channels)` or `(batch_size, height, width,
      channels)`.
    size: Tuple of `(height, width)` integer. Target size.
    interpolation: String, interpolation to use for resizing. Defaults to
      `'bilinear'`. Supports `bilinear`, `nearest`, `bicubic`, `area`,
      `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.

  Returns:
    Array with shape `(size[0], size[1], channels)`. If the input image was a
    NumPy array, the output is a NumPy array, and if it was a TF tensor,
    the output is a TF tensor.
  """
  if len(size) != 2:
    raise ValueError('Expected `size` to be a tuple of 2 integers, '
                     f'but got: {size}.')
  img = tf.convert_to_tensor(x)
  if img.shape.rank is not None:
    if img.shape.rank < 3 or img.shape.rank > 4:
      raise ValueError(
          'Expected an image array with shape `(height, width, channels)`, '
          'or `(batch_size, height, width, channels)`, but '
          f'got input with incorrect rank, of shape {img.shape}.')
  shape = tf.shape(img)
  height, width = shape[-3], shape[-2]
  target_height, target_width = size
  if img.shape.rank is not None:
    static_num_channels = img.shape[-1]
  else:
    static_num_channels = None

  crop_height = tf.cast(
      tf.cast(width * target_height, 'float32') / target_width, 'int32')
  crop_width = tf.cast(
      tf.cast(height * target_width, 'float32') / target_height, 'int32')

  # Set back to input height / width if crop_height / crop_width is not smaller.
  crop_height = tf.minimum(height, crop_height)
  crop_width = tf.minimum(width, crop_width)

  crop_box_hstart = tf.cast(
      tf.cast(height - crop_height, 'float32') / 2, 'int32')
  crop_box_wstart = tf.cast(tf.cast(width - crop_width, 'float32') / 2, 'int32')

  if img.shape.rank == 4:
    crop_box_start = tf.stack([0, crop_box_hstart, crop_box_wstart, 0])
    crop_box_size = tf.stack([-1, crop_height, crop_width, -1])
  else:
    crop_box_start = tf.stack([crop_box_hstart, crop_box_wstart, 0])
    crop_box_size = tf.stack([crop_height, crop_width, -1])

  img = tf.slice(img, crop_box_start, crop_box_size)
  img = tf.image.resize(images=img, size=size, method=interpolation)
  # Apparent bug in resize_images_v2 may cause shape to be lost
  if img.shape.rank is not None:
    if img.shape.rank == 4:
      img.set_shape((None, None, None, static_num_channels))
    if img.shape.rank == 3:
      img.set_shape((None, None, static_num_channels))
  if isinstance(x, np.ndarray):
    return img.numpy()
  return img


def get_interpolation(interpolation):
  interpolation = interpolation.lower()
  if interpolation not in _TF_INTERPOLATION_METHODS:
    raise NotImplementedError(
        'Value not recognized for `interpolation`: {}. Supported values '
        'are: {}'.format(interpolation, _TF_INTERPOLATION_METHODS.keys()))
  return _TF_INTERPOLATION_METHODS[interpolation]


@keras_export('keras.utils.array_to_img',
              'keras.preprocessing.image.array_to_img')
def array_to_img(x, data_format=None, scale=True, dtype=None):
  """Converts a 3D Numpy array to a PIL Image instance.

  Usage:

  ```python
  from PIL import Image
  img = np.random.random(size=(100, 100, 3))
  pil_img = tf.keras.preprocessing.image.array_to_img(img)
  ```


  Args:
      x: Input data, in any form that can be converted to a Numpy array.
      data_format: Image data format, can be either `"channels_first"` or
        `"channels_last"`. Defaults to `None`, in which case the global setting
        `tf.keras.backend.image_data_format()` is used (unless you changed it,
        it defaults to `"channels_last"`).
      scale: Whether to rescale the image such that minimum and maximum values
        are 0 and 255 respectively. Defaults to `True`.
      dtype: Dtype to use. Default to `None`, in which case the global setting
        `tf.keras.backend.floatx()` is used (unless you changed it, it defaults
        to `"float32"`)

  Returns:
      A PIL Image instance.

  Raises:
      ImportError: if PIL is not available.
      ValueError: if invalid `x` or `data_format` is passed.
  """

  if data_format is None:
    data_format = backend.image_data_format()
  if dtype is None:
    dtype = backend.floatx()
  if pil_image is None:
    raise ImportError('Could not import PIL.Image. '
                      'The use of `array_to_img` requires PIL.')
  x = np.asarray(x, dtype=dtype)
  if x.ndim != 3:
    raise ValueError('Expected image array to have rank 3 (single image). '
                     f'Got array with shape: {x.shape}')

  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError(f'Invalid data_format: {data_format}')

  # Original Numpy array x has format (height, width, channel)
  # or (channel, height, width)
  # but target PIL image has format (width, height, channel)
  if data_format == 'channels_first':
    x = x.transpose(1, 2, 0)
  if scale:
    x = x - np.min(x)
    x_max = np.max(x)
    if x_max != 0:
      x /= x_max
    x *= 255
  if x.shape[2] == 4:
    # RGBA
    return pil_image.fromarray(x.astype('uint8'), 'RGBA')
  elif x.shape[2] == 3:
    # RGB
    return pil_image.fromarray(x.astype('uint8'), 'RGB')
  elif x.shape[2] == 1:
    # grayscale
    if np.max(x) > 255:
      # 32-bit signed integer grayscale image. PIL mode "I"
      return pil_image.fromarray(x[:, :, 0].astype('int32'), 'I')
    return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
  else:
    raise ValueError(f'Unsupported channel number: {x.shape[2]}')


@keras_export('keras.utils.img_to_array',
              'keras.preprocessing.image.img_to_array')
def img_to_array(img, data_format=None, dtype=None):
  """Converts a PIL Image instance to a Numpy array.

  Usage:

  ```python
  from PIL import Image
  img_data = np.random.random(size=(100, 100, 3))
  img = tf.keras.preprocessing.image.array_to_img(img_data)
  array = tf.keras.preprocessing.image.img_to_array(img)
  ```


  Args:
      img: Input PIL Image instance.
      data_format: Image data format, can be either `"channels_first"` or
        `"channels_last"`. Defaults to `None`, in which case the global setting
        `tf.keras.backend.image_data_format()` is used (unless you changed it,
        it defaults to `"channels_last"`).
      dtype: Dtype to use. Default to `None`, in which case the global setting
        `tf.keras.backend.floatx()` is used (unless you changed it, it defaults
        to `"float32"`).

  Returns:
      A 3D Numpy array.

  Raises:
      ValueError: if invalid `img` or `data_format` is passed.
  """

  if data_format is None:
    data_format = backend.image_data_format()
  if dtype is None:
    dtype = backend.floatx()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError(f'Unknown data_format: {data_format}')
  # Numpy array x has format (height, width, channel)
  # or (channel, height, width)
  # but original PIL image has format (width, height, channel)
  x = np.asarray(img, dtype=dtype)
  if len(x.shape) == 3:
    if data_format == 'channels_first':
      x = x.transpose(2, 0, 1)
  elif len(x.shape) == 2:
    if data_format == 'channels_first':
      x = x.reshape((1, x.shape[0], x.shape[1]))
    else:
      x = x.reshape((x.shape[0], x.shape[1], 1))
  else:
    raise ValueError(f'Unsupported image shape: {x.shape}')
  return x


@keras_export('keras.utils.save_img', 'keras.preprocessing.image.save_img')
def save_img(path, x, data_format=None, file_format=None, scale=True, **kwargs):
  """Saves an image stored as a Numpy array to a path or file object.

  Args:
      path: Path or file object.
      x: Numpy array.
      data_format: Image data format, either `"channels_first"` or
        `"channels_last"`.
      file_format: Optional file format override. If omitted, the format to use
        is determined from the filename extension. If a file object was used
        instead of a filename, this parameter should always be used.
      scale: Whether to rescale image values to be within `[0, 255]`.
      **kwargs: Additional keyword arguments passed to `PIL.Image.save()`.
  """
  if data_format is None:
    data_format = backend.image_data_format()
  img = array_to_img(x, data_format=data_format, scale=scale)
  if img.mode == 'RGBA' and (file_format == 'jpg' or file_format == 'jpeg'):
    warnings.warn('The JPG format does not support '
                  'RGBA images, converting to RGB.')
    img = img.convert('RGB')
  img.save(path, format=file_format, **kwargs)


@keras_export('keras.utils.load_img', 'keras.preprocessing.image.load_img')
def load_img(path,
             grayscale=False,
             color_mode='rgb',
             target_size=None,
             interpolation='nearest',
             keep_aspect_ratio=False):
  """Loads an image into PIL format.

  Usage:

  ```
  image = tf.keras.preprocessing.image.load_img(image_path)
  input_arr = tf.keras.preprocessing.image.img_to_array(image)
  input_arr = np.array([input_arr])  # Convert single image to a batch.
  predictions = model.predict(input_arr)
  ```

  Args:
      path: Path to image file.
      grayscale: DEPRECATED use `color_mode="grayscale"`.
      color_mode: One of `"grayscale"`, `"rgb"`, `"rgba"`. Default: `"rgb"`.
        The desired image format.
      target_size: Either `None` (default to original size) or tuple of ints
        `(img_height, img_width)`.
      interpolation: Interpolation method used to resample the image if the
        target size is different from that of the loaded image. Supported
        methods are `"nearest"`, `"bilinear"`, and `"bicubic"`. If PIL version
        1.1.3 or newer is installed, `"lanczos"` is also supported. If PIL
        version 3.4.0 or newer is installed, `"box"` and `"hamming"` are also
        supported. By default, `"nearest"` is used.
      keep_aspect_ratio: Boolean, whether to resize images to a target
              size without aspect ratio distortion. The image is cropped in
              the center with target aspect ratio before resizing.

  Returns:
      A PIL Image instance.

  Raises:
      ImportError: if PIL is not available.
      ValueError: if interpolation method is not supported.
  """
  if grayscale:
    warnings.warn('grayscale is deprecated. Please use '
                  'color_mode = "grayscale"')
    color_mode = 'grayscale'
  if pil_image is None:
    raise ImportError('Could not import PIL.Image. '
                      'The use of `load_img` requires PIL.')
  if isinstance(path, io.BytesIO):
    img = pil_image.open(path)
  elif isinstance(path, (pathlib.Path, bytes, str)):
    if isinstance(path, pathlib.Path):
      path = str(path.resolve())
    with open(path, 'rb') as f:
      img = pil_image.open(io.BytesIO(f.read()))
  else:
    raise TypeError('path should be path-like or io.BytesIO'
                    ', not {}'.format(type(path)))

  if color_mode == 'grayscale':
    # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
    # convert it to an 8-bit grayscale image.
    if img.mode not in ('L', 'I;16', 'I'):
      img = img.convert('L')
  elif color_mode == 'rgba':
    if img.mode != 'RGBA':
      img = img.convert('RGBA')
  elif color_mode == 'rgb':
    if img.mode != 'RGB':
      img = img.convert('RGB')
  else:
    raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
  if target_size is not None:
    width_height_tuple = (target_size[1], target_size[0])
    if img.size != width_height_tuple:
      if interpolation not in _PIL_INTERPOLATION_METHODS:
        raise ValueError('Invalid interpolation method {} specified. Supported '
                         'methods are {}'.format(
                             interpolation,
                             ', '.join(_PIL_INTERPOLATION_METHODS.keys())))
      resample = _PIL_INTERPOLATION_METHODS[interpolation]

      if keep_aspect_ratio:
        width, height = img.size
        target_width, target_height = width_height_tuple

        crop_height = (width * target_height) // target_width
        crop_width = (height * target_width) // target_height

        # Set back to input height / width
        # if crop_height / crop_width is not smaller.
        crop_height = min(height, crop_height)
        crop_width = min(width, crop_width)

        crop_box_hstart = (height - crop_height) // 2
        crop_box_wstart = (width - crop_width) // 2
        crop_box_wend = crop_box_wstart + crop_width
        crop_box_hend = crop_box_hstart + crop_height
        crop_box = [
            crop_box_wstart, crop_box_hstart, crop_box_wend, crop_box_hend
        ]
        img = img.resize(width_height_tuple, resample, box=crop_box)
      else:
        img = img.resize(width_height_tuple, resample)
  return img
