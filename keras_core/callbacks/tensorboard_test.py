import collections
import os

import numpy as np
import tensorflow.summary as summary
from tensorflow.compat.v1 import SummaryMetadata
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record

from keras_core import backend
from keras_core import callbacks
from keras_core import losses
from keras_core import models
from keras_core import operations as ops
from keras_core import optimizers
from keras_core import testing
from keras_core.layers import Conv2D
from keras_core.layers import Dense
from keras_core.layers import Embedding
from keras_core.layers import Flatten
from keras_core.layers import Input
from keras_core.layers import Layer
from keras_core.optimizers import schedules

# Note: this file and tensorboard in general has a dependency on tensorflow

# A summary that was emitted during a test. Fields:
#   logdir: str. The logdir of the FileWriter to which the summary was
#     written.
#   tag: str. The name of the summary.
_ObservedSummary = collections.namedtuple("_ObservedSummary", ("logdir", "tag"))


class _SummaryIterator(object):
    """Yields `Event` protocol buffers from a given path."""

    def __init__(self, path):
        self._tf_record_iterator = tf_record.tf_record_iterator(path)

    def __iter__(self):
        return self

    def __next__(self):
        r = next(self._tf_record_iterator)
        return event_pb2.Event.FromString(r)

    next = __next__


class _SummaryFile:
    """A record of summary tags and the files to which they were written.

    Fields `scalars`, `images`, `histograms`, and `tensors` are sets
    containing `_ObservedSummary` values.
    """

    def __init__(self):
        self.scalars = set()
        self.images = set()
        self.histograms = set()
        self.tensors = set()
        self.graph_defs = []
        self.convert_from_v2_summary_proto = False


def get_model_from_layers(model_layers, input_shape, name=None):
    model = models.Sequential(name=name)
    model.add(
        Input(
            input_shape,
            dtype="float32",
        )
    )
    for layer in model_layers:
        model.add(layer)
    return model


def list_summaries(logdir):
    """Read all summaries under the logdir into a `_SummaryFile`.

    Args:
      logdir: A path to a directory that contains zero or more event
        files, either as direct children or in transitive subdirectories.
        Summaries in these events must only contain old-style scalars,
        images, and histograms. Non-summary events, like `graph_def`s, are
        ignored.

    Returns:
      A `_SummaryFile` object reflecting all summaries written to any
      event files in the logdir or any of its descendant directories.

    Raises:
      ValueError: If an event file contains an summary of unexpected kind.
    """
    result = _SummaryFile()
    for dirpath, _, filenames in os.walk(logdir):
        for filename in filenames:
            if not filename.startswith("events.out."):
                continue
            path = os.path.join(dirpath, filename)
            for event in _SummaryIterator(path):
                if event.graph_def:
                    result.graph_defs.append(event.graph_def)
                if not event.summary:  # (e.g., it's a `graph_def` event)
                    continue
                for value in event.summary.value:
                    tag = value.tag
                    # Case on the `value` rather than the summary metadata
                    # because the Keras callback uses `summary_ops_v2` to emit
                    # old-style summaries. See b/124535134.
                    kind = value.WhichOneof("value")
                    container = {
                        "simple_value": result.scalars,
                        "image": result.images,
                        "histo": result.histograms,
                        "tensor": result.tensors,
                    }.get(kind)
                    if container is None:
                        raise ValueError(
                            "Unexpected summary kind %r in event file %s:\n%r"
                            % (kind, path, event)
                        )
                    elif kind == "tensor" and tag != "keras":
                        # Convert the tf2 summary proto to old style for type
                        # checking.
                        plugin_name = value.metadata.plugin_data.plugin_name
                        container = {
                            "images": result.images,
                            "histograms": result.histograms,
                            "scalars": result.scalars,
                        }.get(plugin_name)
                        if container is not None:
                            result.convert_from_v2_summary_proto = True
                        else:
                            container = result.tensors
                    container.add(_ObservedSummary(logdir=dirpath, tag=tag))
    return result


class TestTensorBoardV2(testing.TestCase):
    def setUp(self):
        super(TestTensorBoardV2, self).setUp()
        self.logdir = os.path.join(self.get_temp_dir(), "tb")
        self.train_dir = os.path.join(self.logdir, "train")
        self.validation_dir = os.path.join(self.logdir, "validation")

    def _get_model(self, compile_model=True):
        layers = [
            Conv2D(8, (3, 3)),
            Flatten(),
            Dense(1),
        ]
        model = get_model_from_layers(layers, input_shape=(10, 10, 1))

        if compile_model:
            opt = optimizers.SGD(learning_rate=0.001)
            model.compile(opt, "mse")
        return model

    def test_TensorBoard_default_logdir(self):
        """Regression test for cross-platform pathsep in default logdir."""
        os.chdir(self.get_temp_dir())

        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = callbacks.TensorBoard()  # no logdir specified

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        summary_file = list_summaries(logdir=".")
        train_dir = os.path.join(".", "logs", "train")
        validation_dir = os.path.join(".", "logs", "validation")
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=validation_dir, tag="evaluation_loss_vs_iterations"
                ),
            },
        )

    def test_TensorBoard_basic(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = callbacks.TensorBoard(self.logdir)

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )

    def test_TensorBoard_across_invocations(self):
        """Regression test for summary writer resource use-after-free.

        See: <https://github.com/tensorflow/tensorflow/issues/25707>
        """
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = callbacks.TensorBoard(self.logdir)

        for _ in (1, 2):
            model.fit(
                x,
                y,
                batch_size=2,
                epochs=2,
                validation_data=(x, y),
                callbacks=[tb_cbk],
            )

        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )

    def test_TensorBoard_no_spurious_event_files(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = callbacks.TensorBoard(self.logdir)

        model.fit(x, y, batch_size=2, epochs=2, callbacks=[tb_cbk])

        events_file_run_basenames = set()
        for dirpath, _, filenames in os.walk(self.train_dir):
            if any(fn.startswith("events.out.") for fn in filenames):
                events_file_run_basenames.add(os.path.basename(dirpath))
        self.assertEqual(events_file_run_basenames, {"train"})

    def test_TensorBoard_batch_metrics(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = callbacks.TensorBoard(self.logdir, update_freq=1)

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="batch_loss"),
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )

    def test_TensorBoard_learning_rate_schedules(self):
        model = self._get_model(compile_model=False)
        opt = optimizers.SGD(schedules.CosineDecay(0.01, 1))
        model.compile(opt, "mse")

        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            callbacks=[callbacks.TensorBoard(self.logdir)],
        )

        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.train_dir, tag="epoch_learning_rate"
                ),
            },
        )

    def test_TensorBoard_global_step(self):
        model = self._get_model(compile_model=False)
        opt = optimizers.SGD(schedules.CosineDecay(0.01, 1))
        model.compile(opt, "mse")

        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            verbose=0,
            callbacks=[
                callbacks.TensorBoard(
                    self.logdir,
                    update_freq=1,
                    profile_batch=0,
                    write_steps_per_second=True,
                )
            ],
        )

        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="batch_loss"),
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.train_dir, tag="epoch_learning_rate"
                ),
                _ObservedSummary(
                    logdir=self.train_dir, tag="epoch_steps_per_second"
                ),
                _ObservedSummary(
                    logdir=self.train_dir, tag="batch_steps_per_second"
                ),
            },
        )

    def test_TensorBoard_weight_histograms(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = callbacks.TensorBoard(self.logdir, histogram_freq=1)

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)

        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )
        self.assertEqual(
            self._strip_layer_names(summary_file.histograms, "sequential"),
            {_ObservedSummary(logdir=self.train_dir, tag="histogram")},
        )

    def test_TensorBoard_weight_images(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = callbacks.TensorBoard(
            self.logdir, histogram_freq=1, write_images=True
        )
        model_type = "sequential"

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)

        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )
        self.assertEqual(
            self._strip_layer_names(summary_file.histograms, model_type),
            {
                _ObservedSummary(logdir=self.train_dir, tag="histogram"),
            },
        )
        expected_image_summaries = {
            _ObservedSummary(logdir=self.train_dir, tag="image"),
            _ObservedSummary(logdir=self.train_dir, tag="bias/image"),
            _ObservedSummary(logdir=self.train_dir, tag="kernel/image"),
        }
        self.assertEqual(
            self._strip_variable_names(summary_file.images),
            expected_image_summaries,
        )

    def test_TensorBoard_projector_callback(self):
        layers = [
            Embedding(10, 10, name="test_embedding"),
            Dense(10, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
        model = get_model_from_layers(layers, input_shape=(10,))
        model.compile(
            optimizer="adam", loss=losses.BinaryCrossentropy(from_logits=True)
        )
        x, y = np.ones((10, 10)), np.ones((10, 10))
        tb_cbk = callbacks.TensorBoard(
            self.logdir,
            embeddings_freq=1,
            embeddings_metadata={"test_embedding": "metadata.tsv"},
        )

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        with open(os.path.join(self.logdir, "projector_config.pbtxt")) as f:
            self.assertEqual(
                f.readlines(),
                [
                    "embeddings {\n",
                    "  tensor_name: "
                    '"layer_with_weights-0/embeddings/.ATTRIBUTES/'
                    'VARIABLE_VALUE"\n',
                    '  metadata_path: "metadata.tsv"\n',
                    "}\n",
                ],
            )

    def test_custom_summary(self):
        def scalar_v2_mock(name, data, step=None):
            """A reimplementation of the scalar plugin to avoid circular
            deps."""
            metadata = SummaryMetadata()
            # Should match value in tensorboard/plugins/scalar/metadata.py.
            metadata.plugin_data.plugin_name = "scalars"
            with summary.experimental.summary_scope(
                name, "scalar_summary", values=[data, step]
            ) as (tag, _):
                tensor = ops.convert_to_tensor(data, "float32")
                summary.write(
                    tag=tag,
                    tensor=tensor,
                    step=step,
                    metadata=metadata,
                )

        class LayerWithSummary(Layer):
            def call(self, x):
                scalar_v2_mock("custom_summary", ops.sum(x))
                return x

        model = get_model_from_layers(
            [LayerWithSummary()], input_shape=(5,), name="model"
        )

        model.compile("sgd", "mse", jit_compile=False)  # summary ops can't xla
        tb_cbk = callbacks.TensorBoard(self.logdir, update_freq=1)
        x, y = np.ones((10, 5)), np.ones((10, 5))
        model.fit(
            x, y, batch_size=2, validation_data=(x, y), callbacks=[tb_cbk]
        )
        summary_file = list_summaries(self.logdir)
        # TODO: tensorflow will tag with model/layer_with_summary/custom_summary
        # Jax will only use custom_summary tag
        self.assertEqual(
            self._strip_to_only_final_name(summary_file.scalars),
            {
                _ObservedSummary(logdir=self.train_dir, tag="batch_loss"),
                _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=self.validation_dir, tag="epoch_loss"),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
                _ObservedSummary(
                    logdir=self.train_dir,
                    tag="custom_summary",
                ),
                _ObservedSummary(
                    logdir=self.validation_dir,
                    tag="custom_summary",
                ),
            },
        )
        # self.assertEqual(
        #     summary_file.scalars,
        #     {
        #         _ObservedSummary(logdir=self.train_dir, tag="batch_loss"),
        #         _ObservedSummary(logdir=self.train_dir, tag="epoch_loss"),
        #         _ObservedSummary(logdir=self.validation_dir,
        #               tag="epoch_loss"),
        #         _ObservedSummary(
        #             logdir=self.validation_dir,
        #             tag="evaluation_loss_vs_iterations",
        #         ),
        #         _ObservedSummary(
        #             logdir=self.train_dir,
        #             tag="model/layer_with_summary/custom_summary",
        #         ),
        #         _ObservedSummary(
        #             logdir=self.validation_dir,
        #             tag="model/layer_with_summary/custom_summary",
        #         ),
        #     },
        # )

    def _strip_to_only_final_name(self, summaries):
        """Removes all leading names in a summary

        Args:
            summaries: A `set` of `_ObservedSummary` values.

        Returns:
            A new `set` of `_ObservedSummary` values striped of all
            name except for the terminal one.

        """
        result = set()
        for s in summaries:
            if "/" not in s.tag:
                result.add(s)
            else:
                new_tag = s.tag.split("/")[-1]
                result.add(s._replace(tag=new_tag))
        return result

    def _strip_layer_names(self, summaries, model_type):
        """Deduplicate summary names modulo layer prefix.

        This removes the first slash-component of each tag name: for
        instance, "foo/bar/baz" becomes "bar/baz".

        Args:
            summaries: A `set` of `_ObservedSummary` values.
            model_type: The model type currently being tested.

        Returns:
            A new `set` of `_ObservedSummary` values with layer prefixes
            removed.
        """
        result = set()
        for s in summaries:
            if "/" not in s.tag:
                raise ValueError(f"tag has no layer name: {s.tag!r}")
            start_from = 2 if "subclass" in model_type else 1
            new_tag = "/".join(s.tag.split("/")[start_from:])
            result.add(s._replace(tag=new_tag))
        return result

    def _strip_variable_names(self, summaries):
        """Remove `variable_n` from summary tag

        `variable_n` tag names are added with random numbers. Removing them
        ensures deterministic tag names.

        Args:
            summaries: A `set` of `_ObservedSummary` values.

        Returns:
            A new `set` of `_ObservedSummary` values with layer prefixes
            removed.
        """
        result = set()
        for s in summaries:
            if "/" not in s.tag:
                result.add(s)
            else:
                split_tag = s.tag.split("/")
                if "variable" in split_tag[0]:
                    result.add(s._replace(tag=split_tag[-1]))
                else:
                    result.add(s)
        return result

    def test_TensorBoard_non_blocking(self):
        model = models.Sequential([Dense(1)])
        model.optimizer = optimizers.Adam()
        tb = callbacks.TensorBoard(self.logdir)
        cb_list = callbacks.CallbackList(
            [tb], model=model, epochs=1, steps=100, verbose=0
        )

        tensor = ops.convert_to_tensor(1.0)

        def mock_numpy():
            raise RuntimeError(
                "If this error is seen, TensorBoard is causing a blocking "
                "NumPy conversion."
            )

        tensor.numpy = mock_numpy

        logs = {"metric": tensor}

        cb_list.on_train_begin(logs)
        cb_list.on_epoch_begin(0, logs)
        cb_list.on_train_batch_begin(0, logs)
        cb_list.on_train_batch_end(0, logs)
        cb_list.on_epoch_end(0, logs)
        cb_list.on_train_end(logs)

        cb_list.on_test_begin(logs)
        cb_list.on_test_batch_begin(0, logs)
        cb_list.on_test_batch_end(0, logs)
        cb_list.on_test_end(logs)

        cb_list.on_predict_begin(logs)
        cb_list.on_predict_batch_begin(logs)
        cb_list.on_predict_batch_end(logs)
        cb_list.on_predict_end(logs)


# Note that this test specifies model_type explicitly.
class TestTensorBoardV2NonParameterizedTest(testing.TestCase):
    def setUp(self):
        super(TestTensorBoardV2NonParameterizedTest, self).setUp()
        self.logdir = os.path.join(self.get_temp_dir(), "tb")
        self.train_dir = os.path.join(self.logdir, "train")
        self.validation_dir = os.path.join(self.logdir, "validation")

    def _get_seq_model(self):
        model = models.Sequential(
            [
                Input((10, 10, 1)),
                Conv2D(8, (3, 3)),
                Flatten(),
                Dense(1),
            ]
        )
        opt = optimizers.SGD(learning_rate=0.001)
        model.compile(opt, "mse")
        return model

    def _count_xplane_file(self, logdir):
        profile_dir = os.path.join(logdir, "plugins", "profile")
        count = 0
        for dirpath, dirnames, filenames in os.walk(profile_dir):
            del dirpath  # unused
            del dirnames  # unused
            for filename in filenames:
                if filename.endswith(".xplane.pb"):
                    count += 1
        return count

    def fitModelAndAssertKerasModelWritten(self, model):
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        tb_cbk = callbacks.TensorBoard(
            self.logdir, write_graph=True, profile_batch=0
        )
        model.fit(
            x,
            y,
            batch_size=2,
            epochs=3,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(self.logdir)
        self.assertEqual(
            summary_file.tensors,
            {
                _ObservedSummary(logdir=self.train_dir, tag="keras"),
            },
        )
        if not model.run_eagerly:
            # There should be one train graph
            self.assertLen(summary_file.graph_defs, 1)
            for graph_def in summary_file.graph_defs:
                graph_def_str = str(graph_def)

                # All the model layers should appear in the graphs
                for layer in model.layers:
                    if "input" not in layer.name:
                        self.assertIn(layer.name, graph_def_str)

    def test_TensorBoard_write_sequential_model_no_input_shape(self):
        # TODO: Requires to_json implementation in trainer
        # model = models.Sequential(
        #     [
        #         Conv2D(8, (3, 3)),
        #         Flatten(),
        #         Dense(1),
        #     ]
        # )
        # model.compile("sgd", "mse")
        # self.fitModelAndAssertKerasModelWritten(model)
        pass

    def test_TensorBoard_write_sequential_model_with_input_shape(self):
        # TODO: Requires to_json implementation in trainer
        # model = models.Sequential(
        #     [
        #         Input(input_shape=(10, 10, 1)),
        #         Conv2D(8, (3, 3)),
        #         Flatten(),
        #         Dense(1),
        #     ]
        # )
        # model.compile("sgd", "mse")
        # self.fitModelAndAssertKerasModelWritten(model)
        pass

    def test_TensorBoard_write_model(self):
        # TODO: Requires to_json implementation in trainer
        # See https://github.com/keras-team/keras/blob/ \
        # a8d4a7f1ffc9de3c5932828a107e4e95e8803fb4/ \
        # keras/engine/training.py#L3313
        # inputs = Input([10, 10, 1])
        # x = Conv2D(8, (3, 3), activation="relu")(inputs)
        # x = Flatten()(x)
        # x = Dense(1)(x)
        # model = models.Model(inputs=inputs, outputs=[x])
        # model.compile("sgd", "mse")
        # breakpoint()
        # self.fitModelAndAssertKerasModelWritten(model)
        pass

    def test_TensorBoard_auto_trace(self):
        # TODO: Waiting for implementation for torch/jax for profiling ops
        #if backend.backend() == "jax":
        #       return
        # TODO: Debug profiling for JAX
        # model = self._get_seq_model()
        # x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        # tb_cbk = callbacks.TensorBoard(
        #     self.logdir, histogram_freq=1, profile_batch=1, write_graph=False
        # )

        # model.fit(
        #     x,
        #     y,
        #     batch_size=2,
        #     epochs=2,
        #     validation_data=(x, y),
        #     callbacks=[tb_cbk],
        # )
        # summary_file = list_summaries(self.logdir)

        # self.assertEqual(
        #     summary_file.tensors,
        #     {
        #         _ObservedSummary(logdir=self.train_dir, tag="batch_1"),
        #     },
        # )
        # self.assertEqual(1, self._count_xplane_file(logdir=self.logdir))
        pass
