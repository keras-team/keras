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
"""Public Keras utilities."""

# isort: off

# Serialization related
from keras.saving.legacy.serialization import deserialize_keras_object
from keras.saving.legacy.serialization import serialize_keras_object
from keras.saving.object_registration import CustomObjectScope
from keras.saving.object_registration import custom_object_scope
from keras.saving.object_registration import get_custom_objects
from keras.saving.object_registration import get_registered_name
from keras.saving.object_registration import register_keras_serializable

# Dataset related
from keras.utils.audio_dataset import audio_dataset_from_directory
from keras.utils.text_dataset import text_dataset_from_directory
from keras.utils.timeseries_dataset import timeseries_dataset_from_array
from keras.utils.image_dataset import image_dataset_from_directory
from keras.utils.dataset_utils import split_dataset

# Sequence related
from keras.utils.data_utils import GeneratorEnqueuer
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.data_utils import Sequence
from keras.utils.data_utils import SequenceEnqueuer

# Image related
from keras.utils.image_utils import array_to_img
from keras.utils.image_utils import img_to_array
from keras.utils.image_utils import load_img
from keras.utils.image_utils import save_img

# Python utils
from keras.utils.tf_utils import set_random_seed
from keras.utils.generic_utils import Progbar
from keras.utils.data_utils import get_file

# Preprocessing utils
from keras.utils.feature_space import FeatureSpace

# Internal
from keras.utils.layer_utils import get_source_inputs
from keras.utils.layer_utils import warmstart_embedding_matrix

# Deprecated
from keras.utils.np_utils import normalize
from keras.utils.np_utils import to_categorical
from keras.utils.np_utils import to_ordinal
from keras.utils.data_utils import pad_sequences

# Evaluation related
from keras.utils.sidecar_evaluator import SidecarEvaluator

# Visualization related
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model
