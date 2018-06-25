# About Keras models

There are two main types of models available in Keras: [the Sequential model](/models/sequential), and [the Model class used with the functional API](/models/model).

These models have a number of methods and attributes in common:

- `model.layers` is a flattened list of the layers comprising the model.
- `model.inputs` is the list of input tensors of the model.
- `model.outputs` is the list of output tensors of the model.
- `model.summary()` prints a summary representation of your model. Shortcut for [utils.print_summary](/utils/#print_summary)
- `model.get_config()` returns a dictionary containing the configuration of the model. The model can be reinstantiated from its config via:

```python
config = model.get_config()
model = Model.from_config(config)
# or, for Sequential:
model = Sequential.from_config(config)
```

- `model.get_weights()` returns a list of all weight tensors in the model, as Numpy arrays.
- `model.set_weights(weights)` sets the values of the weights of the model, from a list of Numpy arrays. The arrays in the list should have the same shape as those returned by `get_weights()`.
- `model.to_json()` returns a representation of the model as a JSON string. Note that the representation does not include the weights, only the architecture. You can reinstantiate the same model (with reinitialized weights) from the JSON string via:

```python
from keras.models import model_from_json

json_string = model.to_json()
model = model_from_json(json_string)
```
- `model.to_yaml()` returns a representation of the model as a YAML string. Note that the representation does not include the weights, only the architecture. You can reinstantiate the same model (with reinitialized weights) from the YAML string via:

```python
from keras.models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)
```

- `model.save_weights(filepath)` saves the weights of the model as a HDF5 file.
- `model.load_weights(filepath, by_name=False)` loads the weights of the model from a HDF5 file (created by `save_weights`). By default, the architecture is expected to be unchanged. To load weights into a different architecture (with some layers in common), use `by_name=True` to load only those layers with the same name.

Note: Please also see [How can I install HDF5 or h5py to save my models in Keras?](/getting-started/faq/#how-can-i-install-HDF5-or-h5py-to-save-my-models-in-Keras) in the FAQ for instructions on how to install `h5py`.


## Model subclassing

In addition to these two types of models, you may create your own fully-customizable models by subclassing the `Model` class
and implementing your own forward pass in the `call` method (the `Model` subclassing API was introduced in Keras 2.2.0).

Here's an example of a simple multi-layer perceptron model written as a `Model` subclass:

```python
import keras

class SimpleMLP(keras.Model):

    def __init__(self, use_bn=False, use_dp=False, num_classes=10):
        super(SimpleMLP, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        self.num_classes = num_classes

        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
        if self.use_dp:
            self.dp = keras.layers.Dropout(0.5)
        if self.use_bn:
            self.bn = keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs):
        x = self.dense1(inputs)
        if self.use_dp:
            x = self.dp(x)
        if self.use_bn:
            x = self.bn(x)
        return self.dense2(x)

model = SimpleMLP()
model.compile(...)
model.fit(...)
```

Layers are defined in `__init__(self, ...)`, and the forward pass is specified in `call(self, inputs)`. In `call`, you may specify custom losses by calling `self.add_loss(loss_tensor)` (like you would in a custom layer).

In subclassed models, the model's topology is defined as Python code (rather than as a static graph of layers).
That means the model's topology cannot be inspected or serialized. As a result, the following methods and attributes are **not available for subclassed models**:

- `model.inputs` and `model.outputs`.
- `model.to_yaml()` and `model.to_json()`
- `model.get_config()` and `model.save()`.

**Key point:** use the right API for the job. The `Model` subclassing API can provide you with greater flexbility for implementing complex models,
but it comes at a cost (in addition to these missing features):
it is more verbose, more complex, and has more opportunities for user errors. If possible, prefer using the functional API, which is more user-friendly.
