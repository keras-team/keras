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
"""Central API entry point for v1 versions of summary operations.

This module simply offers a shorter way to access the members of modules
like `tensorboard.plugins.scalar.summary`.
"""

from tensorboard.plugins.audio import summary as _audio_summary
from tensorboard.plugins.custom_scalar import summary as _custom_scalar_summary
from tensorboard.plugins.histogram import summary as _histogram_summary
from tensorboard.plugins.image import summary as _image_summary
from tensorboard.plugins.pr_curve import summary as _pr_curve_summary
from tensorboard.plugins.scalar import summary as _scalar_summary
from tensorboard.plugins.text import summary as _text_summary


audio = _audio_summary.op
audio_pb = _audio_summary.pb

custom_scalar = _custom_scalar_summary.op
custom_scalar_pb = _custom_scalar_summary.pb

histogram = _histogram_summary.op
histogram_pb = _histogram_summary.pb

image = _image_summary.op
image_pb = _image_summary.pb

pr_curve = _pr_curve_summary.op
pr_curve_pb = _pr_curve_summary.pb
pr_curve_streaming_op = _pr_curve_summary.streaming_op
pr_curve_raw_data_op = _pr_curve_summary.raw_data_op
pr_curve_raw_data_pb = _pr_curve_summary.raw_data_pb

scalar = _scalar_summary.op
scalar_pb = _scalar_summary.pb

text = _text_summary.op
text_pb = _text_summary.pb
