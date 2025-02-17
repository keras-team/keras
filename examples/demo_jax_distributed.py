# To run this demo, you will need to spin up a "TPU VM" on Google Cloud.
# Please follow instructions here: https://cloud.google.com/tpu/docs/run-calculation-jax

# Force a JAX backend
import os, pprint, collections

os.environ["KERAS_BACKEND"] = "jax"

pp = pprint.PrettyPrinter()

import jax
import jax.numpy as jnp
import tensorflow as tf  # just for tf.data
import keras  # Keras multi-backend

import numpy as np
from tqdm import tqdm

from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

""" Dataset
Classic MNIST, loaded using tf.data
"""

BATCH_SIZE = 192

(
    (x_train, train_labels),
    (x_eval, eval_labels),
) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype(
    np.float32
)  # from 28x28 to 28x28 x 1 color channel (B&W)
x_eval = np.expand_dims(x_eval, axis=-1).astype(np.float32)

train_data = tf.data.Dataset.from_tensor_slices((x_train, train_labels))
train_data = train_data.shuffle(5000, reshuffle_each_iteration=True)
train_data = train_data.batch(BATCH_SIZE, drop_remainder=True)
train_data = train_data.repeat()

eval_data = tf.data.Dataset.from_tensor_slices((x_eval, eval_labels))
eval_data = eval_data.batch(10000)  # everything as one batch

STEPS_PER_EPOCH = len(train_labels) // BATCH_SIZE

""" Keras model
Simple but non-trivial model with:
* Batch Normalization (non-trainable state updated during training, different training-time and inference behavior)
* Dropout (randomness, different training time and inference behavior)
"""


# Keras "sequential" model building style
def make_backbone():
    return keras.Sequential(
        [
            keras.layers.Rescaling(
                1.0 / 255.0
            ),  # input images are in the range [0, 255]
            keras.layers.Conv2D(
                filters=12, kernel_size=3, padding="same", use_bias=False
            ),
            keras.layers.BatchNormalization(scale=False, center=True),
            keras.layers.Activation("relu"),
            keras.layers.Conv2D(
                filters=24,
                kernel_size=6,
                padding="same",
                use_bias=False,
                strides=2,
            ),
            keras.layers.BatchNormalization(scale=False, center=True),
            keras.layers.Activation("relu"),
            keras.layers.Conv2D(
                filters=32,
                kernel_size=6,
                padding="same",
                use_bias=False,
                strides=2,
                name="large_k",
            ),
            keras.layers.BatchNormalization(scale=False, center=True),
            keras.layers.Activation("relu"),
        ],
        name="backbone",
    )


def make_model():
    input = keras.Input(shape=[28, 28, 1])
    y = make_backbone()(input)
    y = keras.layers.Flatten()(y)
    y = keras.layers.Dense(200, activation="relu")(y)
    y = keras.layers.Dropout(0.4)(y)
    y = keras.layers.Dense(10, activation="softmax")(y)
    model = keras.Model(inputs=input, outputs=y)
    return model


""" JAX-native distribution with a Keras model
For now, you have to write a custom training loop for this
Note: The features required by jax.sharding are not supported by the Colab TPU
runtime at this time, but are available on Cloud TPU VMs and Kaggle TPU VMs.
"""

if len(jax.local_devices()) < 8:
    raise Exception("This part requires 8 devices to run")
else:
    print("\nIdentified local devices:")
    pp.pprint(jax.local_devices())

# ----------------- Keras ---------------------

# instantiate the model
model = make_model()

# learning rate
lr = keras.optimizers.schedules.ExponentialDecay(0.01, STEPS_PER_EPOCH, 0.6)

# optimizer
optimizer = keras.optimizers.Adam(lr)

# initialize all state with .build()
(one_batch, one_batch_labels) = next(iter(train_data))
model.build(one_batch)
optimizer.build(model.trainable_variables)

""" Distribution settings

* Sharding the data on the batch axis
* Replicating all model variables

Note: this implements standard "data parallel" distributed training

* Just for show, sharding the largest convolutional kernel along the
  "channels" axis 4-ways and replicating 2-ways

Note: this does not reflect a best practice but is intended to show
      that you can split a very large kernel across multiple devices
      if you have to
"""

print(
    "\nMostly data-parallel distribution. "
    "Data is sharded across devices while the model is replicated. "
    "For demo purposes, we split the largest kernel 4-ways "
    "(and replicate 2-ways since we have 8 devices)."
)

# ------------------ Jax ----------------------

devices = mesh_utils.create_device_mesh((8,))

# data will be split along the batch axis
data_mesh = Mesh(devices, axis_names=("batch",))  # naming axes of the mesh
# naming axes of the sharded partition
data_sharding = NamedSharding(
    data_mesh,
    P(
        "batch",
    ),
)
# all variables will be replicated on all devices
var_mesh = Mesh(devices, axis_names=("_"))
# in NamedSharding, axes that are not mentioned are replicated (all axes here)
var_replication = NamedSharding(var_mesh, P())

# for the demo, we will split the largest kernel 4-ways (and replicate 2-ways since we have 8 devices)
large_kernel_mesh = Mesh(
    devices.reshape((-1, 4)), axis_names=(None, "out_chan")
)  # naming axes of the mesh
large_kernel_sharding = NamedSharding(
    large_kernel_mesh, P(None, None, None, "out_chan")
)  # naming axes of the sharded partition

# ----------------- Keras ---------------------

# Use Keras APIs to find the variable of a specific layer (we will be sharding this one in a special way)
# In a Conv2D or Dense layer, the variables are 'kernel' and 'bias'
special_layer_var = model.get_layer("backbone").get_layer("large_k").kernel

# ------------------ Jax ----------------------
# - accessing variables in Keras lists model.trainable_variables,
# - model.non_trainable_variables and optimizer.variables

# Apply the distribution settings to the model variables
non_trainable_variables = jax.device_put(
    model.non_trainable_variables, var_replication
)
optimizer_variables = jax.device_put(optimizer.variables, var_replication)
# this is what you would do replicate all trainable variables:
# trainable_variables = jax.device_put(model.trainable_variables, var_replication)

# For the demo, we split the largest kernel 4-ways instead of replicating it.
# We still replicate all other trainable variables as in standard "data-parallel"
# distributed training.
print_once = True
trainable_variables = model.trainable_variables
for i, v in enumerate(trainable_variables):
    if v is special_layer_var:
        # Apply distribution settings: sharding
        sharded_v = jax.device_put(v, large_kernel_sharding)
        trainable_variables[i] = sharded_v

        print("Sharding of convolutional", v.name, v.shape)
        jax.debug.visualize_array_sharding(
            jnp.reshape(sharded_v, [-1, v.shape[-1]])
        )
    else:
        # Apply distribution settings: replication
        replicated_v = jax.device_put(v, var_replication)
        trainable_variables[i] = replicated_v

        if print_once:
            print_once = False
            print(
                "\nSharding of all other model variables (they are replicated)"
            )
            jax.debug.visualize_array_sharding(
                jnp.reshape(replicated_v, [-1, v.shape[-1]])
            )

# collect state in a handy named tuple
TrainingState = collections.namedtuple(
    "TrainingState",
    ["trainable_variables", "non_trainable_variables", "optimizer_variables"],
)
device_train_state = TrainingState(
    trainable_variables=trainable_variables,
    non_trainable_variables=non_trainable_variables,
    optimizer_variables=optimizer_variables,
)
# display data sharding
x, y = next(iter(train_data))
sharded_x = jax.device_put(x.numpy(), data_sharding)
print("Data sharding")
jax.debug.visualize_array_sharding(jnp.reshape(sharded_x, [-1, 28 * 28]))

# ------------------ Jax ----------------------
# - Using Keras-provided stateless APIs
# - model.stateless_call
# - optimizer.stateless_apply
# These functions also work on other backends.

# define loss
loss = keras.losses.SparseCategoricalCrossentropy()


# This is the loss function that will be differentiated.
# Keras provides a pure functional forward pass: model.stateless_call
def compute_loss(trainable_variables, non_trainable_variables, x, y):
    y_pred, updated_non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x
    )
    loss_value = loss(y, y_pred)
    return loss_value, updated_non_trainable_variables


# function to compute gradients
compute_gradients = jax.value_and_grad(compute_loss, has_aux=True)


# Training step: Keras provides a pure functional optimizer.stateless_apply
@jax.jit
def train_step(train_state, x, y):
    (loss_value, non_trainable_variables), grads = compute_gradients(
        train_state.trainable_variables,
        train_state.non_trainable_variables,
        x,
        y,
    )

    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        train_state.optimizer_variables, grads, train_state.trainable_variables
    )

    return loss_value, TrainingState(
        trainable_variables, non_trainable_variables, optimizer_variables
    )


# training loop
EPOCHS = 5
print("\nTraining:")
data_iter = iter(train_data)
for epoch in range(EPOCHS):
    loss_value = None  # default
    for i in tqdm(range(STEPS_PER_EPOCH)):
        x, y = next(data_iter)
        sharded_x = jax.device_put(x.numpy(), data_sharding)
        loss_value, device_train_state = train_step(
            device_train_state, sharded_x, y.numpy()
        )
    print("Epoch", epoch, "loss:", loss_value)

# The output of the model is still sharded. Sharding follows the data.

data, labels = next(iter(eval_data))
sharded_data = jax.device_put(data.numpy(), data_sharding)


@jax.jit
def predict(data):
    predictions, updated_non_trainable_variables = model.stateless_call(
        device_train_state.trainable_variables,
        device_train_state.non_trainable_variables,
        data,
    )
    return predictions


predictions = predict(sharded_data)
print("\nModel output sharding follows data sharding:")
jax.debug.visualize_array_sharding(predictions)

# Post-processing model state update to write them back into the model
update = lambda variable, value: variable.assign(value)

jax.tree_map(
    update, model.trainable_variables, device_train_state.trainable_variables
)
jax.tree_map(
    update,
    model.non_trainable_variables,
    device_train_state.non_trainable_variables,
)
jax.tree_map(
    update, optimizer.variables, device_train_state.optimizer_variables
)

# check that the model has the new state by running an eval
# known issue: the optimizer should not be required here
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
print("\nUpdating model and running an eval:")
loss, accuracy = model.evaluate(eval_data)
print("The model achieved an evaluation accuracy of:", accuracy)
