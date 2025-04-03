# Copyright 2024 The etils Authors.
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

"""Utils for colab/jupyter.

Usage:

```python
from etils.ecolab import array_as_img
```

"""

from __future__ import annotations

import functools
import traceback
from typing import Any, Optional, Tuple

from etils import enp
from etils import epy
from etils.epy import _internal

with _internal.check_missing_deps():
  # pylint: disable=g-import-not-at-top
  import IPython
  import IPython.display
  # pylint: enable=g-import-not-at-top


with epy.lazy_imports(
    error_callback='Use `pip install etils[ecolab]` to install missing deps'
):
  # pylint: disable=g-import-not-at-top
  import mediapy as media
  # pylint: enable=g-import-not-at-top


Array = Any

# Images smaller than this are displayed as text (e.g. `np.array([...`)
_MIN_IMG_SHAPE: Tuple[int, int] = (10, 10)


def show(*objs, **kwargs) -> None:
  """Alias for `IPython.display.display`."""
  IPython.display.display(*objs, **kwargs)


def auto_plot_array(
    *,
    # If updating this, also update `_array_repr_html_inner` !!!
    video_min_num_frames: int = 15,
    # Images outside this range are rescalled
    height: None | int | tuple[int, int] = (100, 250),
    show_images_kwargs: Optional[dict[str, Any]] = None,
    show_videos_kwargs: Optional[dict[str, Any]] = None,
) -> None:
  """If called, 2d/3d imgage arrays will be plotted as images in colab/jupyter.

  Usage:

  >>> ecolab.auto_plot_array()
  >>> np.zeros((28, 28, 3))  # Displayed as image

  Args:
    video_min_num_frames: Video `(num_frames, h, w, c)` with less than this
      number of frames will be displayed as individual images
    height: `(min, max)` image height in pixels. Images smaller/larger will be
      reshaped. `None` to disable. If a single number, assume `min == max`.
    show_images_kwargs: Kwargs forwarded to `mediapy.show_images`
    show_videos_kwargs: Kwargs forwarded to `mediapy.show_videos`
  """

  ipython = IPython.get_ipython()
  if ipython is None:
    return  # Non-notebook environement

  array_repr_html_fn = functools.partial(
      array_repr_html,
      video_min_num_frames=video_min_num_frames,
      height=height,
      show_images_kwargs=show_images_kwargs,
      show_videos_kwargs=show_videos_kwargs,
  )

  # Register the new representation fo np, tf and jax array
  print('Display big np/tf/jax arrays as image for nicer IPython display')
  formatter = ipython.display_formatter.formatters['text/html']

  # TODO(epot): How to support lazy-imports without catching everything ?
  # Try registering jax
  try:
    jnp = enp.lazy.jnp
  except ImportError:
    pass
  else:
    # The array type is not exposed in the public API (registering jnp.ndarray
    # does not works), so dynamically extracting the type
    jax_array_cls = type(jnp.zeros(shape=()))  # DeviceArrayBase
    formatter.for_type(jax_array_cls, array_repr_html_fn)

  # Try registering TF
  try:
    tf = enp.lazy.tf
  except ImportError:
    pass
  else:
    formatter.for_type(tf.Tensor, array_repr_html_fn)

  # Try registering Torch
  try:
    torch = enp.lazy.torch
  except ImportError:
    pass
  else:
    formatter.for_type(torch.Tensor, array_repr_html_fn)

  # Register np
  formatter.for_type(enp.lazy.np.ndarray, array_repr_html_fn)


def array_repr_html(
    array: Array,
    **kwargs: Any,
) -> Optional[str]:
  """Returns the HTML `<img/>` repr, or `None` if array is not an image."""
  try:
    return _array_repr_html_inner(array, **kwargs)
  except Exception:
    # IPython display silence exceptions, so display it here
    traceback.print_exc()
    raise


def _array_repr_html_inner(
    img: Array,
    *,
    # If updating this, also update `auto_plot_array` !!!
    video_min_num_frames: int = 15,
    height: None | int | tuple[int, int] = (100, 250),
    show_images_kwargs: Optional[dict[str, Any]] = None,
    show_videos_kwargs: Optional[dict[str, Any]] = None,
) -> Optional[str]:
  """Display the normalized img, or `None` if the input is not an image."""
  show_images_kwargs = show_images_kwargs or {}
  show_videos_kwargs = show_videos_kwargs or {}

  if not enp.lazy.is_array(img):  # Not an array
    return None

  # Normalize tf.Tensor into np.array
  if enp.lazy.is_tf(img) or enp.lazy.is_torch(img):
    img = img.numpy()

  shape = img.shape
  ndim = len(shape)

  # Infer the array type (image or video ?)
  if ndim == 2:
    img_shape = shape
    num_channel = 1
  elif ndim == 3:
    img_shape = shape[:2]
    num_channel = shape[-1]
  elif ndim == 4:
    img_shape = shape[1:3]
    num_channel = shape[-1]
    num_frames = shape[0]
  else:
    return None

  # Filter non-images
  if 0 in shape:  # Empty image
    return None
  if _smaller_than(img_shape, _MIN_IMG_SHAPE):
    return None
  if num_channel not in {1, 3, 4}:
    return None

  show_images_kwargs = show_images_kwargs.copy()
  show_videos_kwargs = show_videos_kwargs.copy()

  # Resize small/large images to X pixels (otherwise, difficult to see)
  if height:
    if isinstance(height, int):
      min_height = height
      max_height = height
    else:
      min_height, max_height = height
    del height
    target_height = img_shape[0]  # (h, w)
    target_height = max(target_height, min_height)
    target_height = min(target_height, max_height)

    show_images_kwargs.setdefault('height', target_height)
    show_videos_kwargs.setdefault('height', target_height)

  if ndim < 4:
    out = media.show_image(img, return_html=True, **show_images_kwargs)
  elif num_frames < video_min_num_frames:
    out = media.show_images(img, return_html=True, **show_images_kwargs)
  else:
    # TODO(epot): media.show_video does not support single channel video
    if num_channel != 3:
      return None
    # Dynamically compute the frame-rate, capped at 25 FPS
    fps = min(num_frames // 5, 25.0)

    show_videos_kwargs.setdefault('fps', fps)

    out = media.show_video(
        img,
        return_html=True,
        **show_videos_kwargs,
    )
  return out


def _smaller_than(shape: tuple[int, ...], min_shape: tuple[int, ...]) -> bool:
  """Returns True if one of the dim of `shape` is smaller than `min_shape`."""
  return any(dim < min_dim for dim, min_dim in zip(shape, min_shape))
