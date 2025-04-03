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
"""Public APIs for the HParams plugin.

This module supports a spectrum of use cases, depending on how much
structure you want. In the simplest case, you can simply collect your
hparams into a dict, and use a Keras callback to record them:

>>> from tensorboard.plugins.hparams import api as hp
>>> hparams = {
...     "optimizer": "adam",
...     "fc_dropout": 0.2,
...     "neurons": 128,
...     # ...
... }
>>>
>>> model = model_fn(hparams)
>>> callbacks = [
>>>     tf.keras.callbacks.TensorBoard(logdir),
>>>     hp.KerasCallback(logdir, hparams),
>>> ]
>>> model.fit(..., callbacks=callbacks)

The Keras callback requires that TensorFlow eager execution be enabled.

If not using Keras, use the `hparams` function to write the values
directly:

>>> # In eager mode:
>>> with tf.create_file_writer(logdir).as_default():
...   hp.hparams(hparams)
>>>
>>> # In legacy graph mode:
>>> with tf.compat.v2.create_file_writer(logdir).as_default() as w:
...   sess.run(w.init())
...   sess.run(hp.hparams(hparams))
...   sess.run(w.flush())

To control how hyperparameters and metrics appear in the TensorBoard UI,
you can define `HParam` and `Metric` objects, and write an experiment
summary to the top-level log directory:

>>> HP_OPTIMIZER = hp.HParam("optimizer")
>>> HP_FC_DROPOUT = hp.HParam(
...     "fc_dropout",
...     display_name="f.c. dropout",
...     description="Dropout rate for fully connected subnet.",
... )
>>> HP_NEURONS = hp.HParam("neurons", description="Neurons per dense layer")
>>>
>>> with tf.summary.create_file_writer(base_logdir).as_default():
...   hp.hparams_config(
...       hparams=[
...           HP_OPTIMIZER,
...           HP_FC_DROPOUT,
...           HP_NEURONS,
...       ],
...       metrics=[
...           hp.Metric("xent", group="validation", display_name="cross-entropy"),
...           hp.Metric("f1", group="validation", display_name="F&#x2081; score"),
...           hp.Metric("loss", group="train", display_name="training loss"),
...       ],
...   )

You can continue to pass a string-keyed dict to the Keras callback or
the `hparams` function, or you can use `HParam` objects as the keys. The
latter approach enables better static analysis: your favorite Python
linter can tell you if you misspell a hyperparameter name, your IDE can
help you find all the places where a hyperparameter is used, etc:

>>> hparams = {
...     HP_OPTIMIZER: "adam",
...     HP_FC_DROPOUT: 0.2,
...     HP_NEURONS: 128,
...     # ...
... }
>>>
>>> model = model_fn(hparams)
>>> callbacks = [
>>>     tf.keras.callbacks.TensorBoard(logdir),
>>>     hp.KerasCallback(logdir, hparams),
>>> ]

Finally, you can choose to annotate your hparam definitions with domain
information:

>>> HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam", "sgd"]))
>>> HP_FC_DROPOUT = hp.HParam("fc_dropout", hp.RealInterval(0.1, 0.4))
>>> HP_NEURONS = hp.HParam("neurons", hp.IntInterval(64, 256))

The TensorBoard HParams plugin does not provide tuners, but you can
integrate these domains into your preferred tuning framework if you so
desire. The domains will also be reflected in the TensorBoard UI.

See the `Experiment`, `HParam`, `Metric`, and `KerasCallback` classes
for API specifications. Consult the `hparams_demo.py` script in the
TensorBoard repository for an end-to-end MNIST example.
"""


from tensorboard.plugins.hparams import _keras
from tensorboard.plugins.hparams import summary_v2


Discrete = summary_v2.Discrete
Domain = summary_v2.Domain
HParam = summary_v2.HParam
IntInterval = summary_v2.IntInterval
Metric = summary_v2.Metric
RealInterval = summary_v2.RealInterval
hparams = summary_v2.hparams
hparams_pb = summary_v2.hparams_pb
hparams_config = summary_v2.hparams_config
hparams_config_pb = summary_v2.hparams_config_pb

KerasCallback = _keras.Callback


del _keras
del summary_v2
