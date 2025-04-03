# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Summary creation methods for the HParams plugin.

Typical usage for exporting summaries in a hyperparameters-tuning experiment:
1. Create the experiment (once) by calling experiment_pb() and exporting
   the resulting summary into a top-level (empty) run.
2. In each training session in the experiment, call session_start_pb() before
   the session starts, exporting the resulting summary into a uniquely named
   run for the session, say <session_name>.
3. Train the model in the session, exporting each metric as a scalar summary
   in runs of the form <session_name>/<sub_dir>, where <sub_dir> can be empty a
   (in which case the run is just the <session_name>) and depends on the
   metric. The name of such a metric is a (group, tag) pair given by
   (<sub_dir>, tag) where tag is the tag of the scalar summary.
   When calling experiment_pb in step 1, you'll need to pass all the metric
   names used in the experiemnt.
4. When the session completes, call session_end_pb() and export the resulting
   summary into the same session run <session_name>.
"""


import time


import tensorflow as tf

from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import plugin_data_pb2


def experiment_pb(
    hparam_infos, metric_infos, user="", description="", time_created_secs=None
):
    """Creates a summary that defines a hyperparameter-tuning experiment.

    Args:
      hparam_infos: Array of api_pb2.HParamInfo messages. Describes the
          hyperparameters used in the experiment.
      metric_infos: Array of api_pb2.MetricInfo messages. Describes the metrics
          used in the experiment. See the documentation at the top of this file
          for how to populate this.
      user: String. An id for the user running the experiment
      description: String. A description for the experiment. May contain markdown.
      time_created_secs: float. The time the experiment is created in seconds
      since the UNIX epoch. If None uses the current time.

    Returns:
      A summary protobuffer containing the experiment definition.
    """
    if time_created_secs is None:
        time_created_secs = time.time()
    experiment = api_pb2.Experiment(
        description=description,
        user=user,
        time_created_secs=time_created_secs,
        hparam_infos=hparam_infos,
        metric_infos=metric_infos,
    )
    return _summary(
        metadata.EXPERIMENT_TAG,
        plugin_data_pb2.HParamsPluginData(experiment=experiment),
    )


def session_start_pb(
    hparams, model_uri="", monitor_url="", group_name="", start_time_secs=None
):
    """Constructs a SessionStartInfo protobuffer.

    Creates a summary that contains a training session metadata information.
    One such summary per training session should be created. Each should have
    a different run.

    Args:
      hparams: A dictionary with string keys. Describes the hyperparameter values
               used in the session, mapping each hyperparameter name to its value.
               Supported value types are  `bool`, `int`, `float`, `str`, `list`,
               `tuple`.
               The type of value must correspond to the type of hyperparameter
               (defined in the corresponding api_pb2.HParamInfo member of the
               Experiment protobuf) as follows:

                +-----------------+---------------------------------+
                |Hyperparameter   | Allowed (Python) value types    |
                |type             |                                 |
                +-----------------+---------------------------------+
                |DATA_TYPE_BOOL   | bool                            |
                |DATA_TYPE_FLOAT64| int, float                      |
                |DATA_TYPE_STRING | str, tuple, list   |
                +-----------------+---------------------------------+

               Tuple and list instances will be converted to their string
               representation.
      model_uri: See the comment for the field with the same name of
                 plugin_data_pb2.SessionStartInfo.
      monitor_url: See the comment for the field with the same name of
                   plugin_data_pb2.SessionStartInfo.
      group_name:  See the comment for the field with the same name of
                   plugin_data_pb2.SessionStartInfo.
      start_time_secs: float. The time to use as the session start time.
                       Represented as seconds since the UNIX epoch. If None uses
                       the current time.
    Returns:
      The summary protobuffer mentioned above.
    """
    if start_time_secs is None:
        start_time_secs = time.time()
    session_start_info = plugin_data_pb2.SessionStartInfo(
        model_uri=model_uri,
        monitor_url=monitor_url,
        group_name=group_name,
        start_time_secs=start_time_secs,
    )
    for hp_name, hp_val in hparams.items():
        # Boolean typed values need to be checked before integers since in Python
        # isinstance(True/False, int) returns True.
        if isinstance(hp_val, bool):
            session_start_info.hparams[hp_name].bool_value = hp_val
        elif isinstance(hp_val, (float, int)):
            session_start_info.hparams[hp_name].number_value = hp_val
        elif isinstance(hp_val, str):
            session_start_info.hparams[hp_name].string_value = hp_val
        elif isinstance(hp_val, (list, tuple)):
            session_start_info.hparams[hp_name].string_value = str(hp_val)
        else:
            raise TypeError(
                "hparams[%s]=%s has type: %s which is not supported"
                % (hp_name, hp_val, type(hp_val))
            )
    return _summary(
        metadata.SESSION_START_INFO_TAG,
        plugin_data_pb2.HParamsPluginData(
            session_start_info=session_start_info
        ),
    )


def session_end_pb(status, end_time_secs=None):
    """Constructs a SessionEndInfo protobuffer.

    Creates a summary that contains status information for a completed
    training session. Should be exported after the training session is completed.
    One such summary per training session should be created. Each should have
    a different run.

    Args:
      status: A tensorboard.hparams.Status enumeration value denoting the
          status of the session.
      end_time_secs: float. The time to use as the session end time. Represented
          as seconds since the unix epoch. If None uses the current time.

    Returns:
      The summary protobuffer mentioned above.
    """
    if end_time_secs is None:
        end_time_secs = time.time()

    session_end_info = plugin_data_pb2.SessionEndInfo(
        status=status, end_time_secs=end_time_secs
    )
    return _summary(
        metadata.SESSION_END_INFO_TAG,
        plugin_data_pb2.HParamsPluginData(session_end_info=session_end_info),
    )


def _summary(tag, hparams_plugin_data):
    """Returns a summary holding the given HParamsPluginData message.

    Helper function.

    Args:
      tag: string. The tag to use.
      hparams_plugin_data: The HParamsPluginData message to use.
    """
    summary = tf.compat.v1.Summary()
    tb_metadata = metadata.create_summary_metadata(hparams_plugin_data)
    raw_metadata = tb_metadata.SerializeToString()
    tf_metadata = tf.compat.v1.SummaryMetadata.FromString(raw_metadata)
    summary.value.add(
        tag=tag,
        metadata=tf_metadata,
        tensor=_TF_NULL_TENSOR,
    )
    return summary


# Like `metadata.NULL_TENSOR`, but with the TensorFlow version of the
# proto. Slight kludge needed to expose the `TensorProto` type.
_TF_NULL_TENSOR = type(tf.make_tensor_proto(0)).FromString(
    metadata.NULL_TENSOR.SerializeToString()
)
