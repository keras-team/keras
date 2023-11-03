"""
Title: Distilling Vision Transformers
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2022/04/05
Last modified: 2023/09/16
Description: Distillation of Vision Transformers through attention.
Accelerator: GPU
"""
"""
## Introduction

In the original *Vision Transformers* (ViT) paper
([Dosovitskiy et al.](https://arxiv.org/abs/2010.11929)),
the authors concluded that to perform on par with Convolutional Neural Networks (CNNs),
ViTs need to be pre-trained on larger datasets. The larger the better. This is mainly
due to the lack of inductive biases in the ViT architecture -- unlike CNNs,
they don't have layers that exploit locality. In a follow-up paper
([Steiner et al.](https://arxiv.org/abs/2106.10270)),
the authors show that it is possible to substantially improve the performance of ViTs
with stronger regularization and longer training.

Many groups have proposed different ways to deal with the problem
of data-intensiveness of ViT training.
One such way was shown in the *Data-efficient image Transformers*,
(DeiT) paper ([Touvron et al.](https://arxiv.org/abs/2012.12877)). The
authors introduced a distillation technique that is specific to transformer-based vision
models. DeiT is among the first works to show that it's possible to train ViTs well
without using larger datasets.

In this example, we implement the distillation recipe proposed in DeiT. This
requires us to slightly tweak the original ViT architecture and write a custom training
loop to implement the distillation recipe.

To comfortably navigate through this example, you'll be expected to know how a ViT and
knowledge distillation work. The following are good resources in case you needed a
refresher:

* [ViT on keras.io](https://keras.io/examples/vision/image_classification_with_vision_transformer)
* [Knowledge distillation on keras.io](https://keras.io/examples/vision/knowledge_distillation/)
"""

"""
## Imports
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers

tfds.disable_progress_bar()
keras.utils.set_random_seed(42)

"""
## Constants
"""

# Model
MODEL_TYPE = "deit_distilled_tiny_patch16_224"
RESOLUTION = 224
PATCH_SIZE = 16
NUM_PATCHES = (RESOLUTION // PATCH_SIZE) ** 2
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 192
NUM_HEADS = 3
NUM_LAYERS = 12
MLP_UNITS = [
    PROJECTION_DIM * 4,
    PROJECTION_DIM,
]
DROPOUT_RATE = 0.0
DROP_PATH_RATE = 0.1

# Training
NUM_EPOCHS = 20
BASE_LR = 0.0005
WEIGHT_DECAY = 0.0001

# Data
BATCH_SIZE = 256
AUTO = tf.data.AUTOTUNE
NUM_CLASSES = 5

"""
You probably noticed that `DROPOUT_RATE` has been set 0.0. Dropout has been used
in the implementation to keep it complete. For smaller models (like the one used in
this example), you don't need it, but for bigger models, using dropout helps.
"""

"""
## Load the `tf_flowers` dataset and prepare preprocessing utilities

The authors use an array of different augmentation techniques, including MixUp
([Zhang et al.](https://arxiv.org/abs/1710.09412)),
RandAugment ([Cubuk et al.](https://arxiv.org/abs/1909.13719)),
and so on. However, to keep the example simple to work through, we'll discard them.
"""


def preprocess_dataset(is_training=True):
    def fn(image, label):
        if is_training:
            # Resize to a bigger spatial resolution and take the random
            # crops.
            image = tf.image.resize(image, (RESOLUTION + 20, RESOLUTION + 20))
            image = tf.image.random_crop(image, (RESOLUTION, RESOLUTION, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, (RESOLUTION, RESOLUTION))
        label = tf.one_hot(label, depth=NUM_CLASSES)
        return image, label

    return fn


def prepare_dataset(dataset, is_training=True):
    if is_training:
        dataset = dataset.shuffle(BATCH_SIZE * 10)
    dataset = dataset.map(
        preprocess_dataset(is_training), num_parallel_calls=AUTO
    )
    return dataset.batch(BATCH_SIZE).prefetch(AUTO)


train_dataset, val_dataset = tfds.load(
    "tf_flowers", split=["train[:90%]", "train[90%:]"], as_supervised=True
)
num_train = train_dataset.cardinality()
num_val = val_dataset.cardinality()
print(f"Number of training examples: {num_train}")
print(f"Number of validation examples: {num_val}")

train_dataset = prepare_dataset(train_dataset, is_training=True)
val_dataset = prepare_dataset(val_dataset, is_training=False)

"""
## Implementing the DeiT variants of ViT

Since DeiT is an extension of ViT it'd make sense to first implement ViT and then extend
it to support DeiT's components.

First, we'll implement a layer for Stochastic Depth
([Huang et al.](https://arxiv.org/abs/1603.09382))
which is used in DeiT for regularization.
"""


# Referred from: github.com:rwightman/pytorch-image-models.
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=True):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


"""
Now, we'll implement the MLP and Transformer blocks.
"""


def mlp(x, dropout_rate: float, hidden_units):
    """FFN for a Transformer block."""
    # Iterate over the hidden units and
    # add Dense => Dropout.
    for idx, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation=tf.nn.gelu if idx == 0 else None,
        )(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer(drop_prob: float, name: str) -> keras.Model:
    """Transformer block with pre-norm."""
    num_patches = (
        NUM_PATCHES + 2 if "distilled" in MODEL_TYPE else NUM_PATCHES + 1
    )
    encoded_patches = layers.Input((num_patches, PROJECTION_DIM))

    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)

    # Multi Head Self Attention layer 1.
    attention_output = layers.MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=PROJECTION_DIM,
        dropout=DROPOUT_RATE,
    )(x1, x1)
    attention_output = (
        StochasticDepth(drop_prob)(attention_output)
        if drop_prob
        else attention_output
    )

    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])

    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)

    # MLP layer 1.
    x4 = mlp(x3, hidden_units=MLP_UNITS, dropout_rate=DROPOUT_RATE)
    x4 = StochasticDepth(drop_prob)(x4) if drop_prob else x4

    # Skip connection 2.
    outputs = layers.Add()([x2, x4])

    return keras.Model(encoded_patches, outputs, name=name)


"""
We'll now implement a `ViTClassifier` class building on top of the components we just
developed. Here we'll be following the original pooling strategy used in the ViT paper --
use a class token and use the feature representations corresponding to it for
classification.
"""


class ViTClassifier(keras.Model):
    """Vision Transformer base class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Patchify + linear projection + reshaping.
        self.projection = keras.Sequential(
            [
                layers.Conv2D(
                    filters=PROJECTION_DIM,
                    kernel_size=(PATCH_SIZE, PATCH_SIZE),
                    strides=(PATCH_SIZE, PATCH_SIZE),
                    padding="VALID",
                    name="conv_projection",
                ),
                layers.Reshape(
                    target_shape=(NUM_PATCHES, PROJECTION_DIM),
                    name="flatten_projection",
                ),
            ],
            name="projection",
        )

        # Positional embedding.
        init_shape = (
            1,
            NUM_PATCHES + 1,
            PROJECTION_DIM,
        )
        self.positional_embedding = tf.Variable(
            tf.zeros(init_shape), name="pos_embedding"
        )

        # Transformer blocks.
        dpr = [x for x in tf.linspace(0.0, DROP_PATH_RATE, NUM_LAYERS)]
        self.transformer_blocks = [
            transformer(drop_prob=dpr[i], name=f"transformer_block_{i}")
            for i in range(NUM_LAYERS)
        ]

        # CLS token.
        initial_value = tf.zeros((1, 1, PROJECTION_DIM))
        self.cls_token = tf.Variable(
            initial_value=initial_value, trainable=True, name="cls"
        )

        # Other layers.
        self.dropout = layers.Dropout(DROPOUT_RATE)
        self.layer_norm = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
        self.head = layers.Dense(
            NUM_CLASSES,
            name="classification_head",
        )

    def call(self, inputs):
        n = tf.shape(inputs)[0]

        # Create patches and project the patches.
        projected_patches = self.projection(inputs)

        # Append class token if needed.
        cls_token = tf.tile(self.cls_token, (n, 1, 1))
        cls_token = tf.cast(cls_token, projected_patches.dtype)
        projected_patches = tf.concat([cls_token, projected_patches], axis=1)

        # Add positional embeddings to the projected patches.
        encoded_patches = (
            self.positional_embedding + projected_patches
        )  # (B, number_patches, projection_dim)
        encoded_patches = self.dropout(encoded_patches)

        # Iterate over the number of layers and stack up blocks of
        # Transformer.
        for transformer_module in self.transformer_blocks:
            # Add a Transformer block.
            encoded_patches = transformer_module(encoded_patches)

        # Final layer normalization.
        representation = self.layer_norm(encoded_patches)

        # Pool representation.
        encoded_patches = representation[:, 0]

        # Classification head.
        output = self.head(encoded_patches)
        return output


"""
This class can be used standalone as ViT and is end-to-end trainable. Just remove the
`distilled` phrase in `MODEL_TYPE` and it should work with `vit_tiny = ViTClassifier()`.
Let's now extend it to DeiT. The following figure presents the schematic of DeiT (taken
from the DeiT paper):

![](https://i.imgur.com/5lmg2Xs.png)

Apart from the class token, DeiT has another token for distillation. During distillation,
the logits corresponding to the class token are compared to the true labels, and the
logits corresponding to the distillation token are compared to the teacher's predictions.
"""


class ViTDistilled(ViTClassifier):
    def __init__(self, regular_training=False, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = 2
        self.regular_training = regular_training

        # CLS and distillation tokens, positional embedding.
        init_value = tf.zeros((1, 1, PROJECTION_DIM))
        self.dist_token = tf.Variable(init_value, name="dist_token")
        self.positional_embedding = tf.Variable(
            tf.zeros(
                (
                    1,
                    NUM_PATCHES + self.num_tokens,
                    PROJECTION_DIM,
                )
            ),
            name="pos_embedding",
        )

        # Head layers.
        self.head = layers.Dense(
            NUM_CLASSES,
            name="classification_head",
        )
        self.head_dist = layers.Dense(
            NUM_CLASSES,
            name="distillation_head",
        )

    def call(self, inputs, training=False):
        n = tf.shape(inputs)[0]

        # Create patches and project the patches.
        projected_patches = self.projection(inputs)

        # Append the tokens.
        cls_token = tf.tile(self.cls_token, (n, 1, 1))
        dist_token = tf.tile(self.dist_token, (n, 1, 1))
        cls_token = tf.cast(cls_token, projected_patches.dtype)
        dist_token = tf.cast(dist_token, projected_patches.dtype)
        projected_patches = tf.concat(
            [cls_token, dist_token, projected_patches], axis=1
        )

        # Add positional embeddings to the projected patches.
        encoded_patches = (
            self.positional_embedding + projected_patches
        )  # (B, number_patches, projection_dim)
        encoded_patches = self.dropout(encoded_patches)

        # Iterate over the number of layers and stack up blocks of
        # Transformer.
        for transformer_module in self.transformer_blocks:
            # Add a Transformer block.
            encoded_patches = transformer_module(encoded_patches)

        # Final layer normalization.
        representation = self.layer_norm(encoded_patches)

        # Classification heads.
        x, x_dist = (
            self.head(representation[:, 0]),
            self.head_dist(representation[:, 1]),
        )

        if not training or self.regular_training:
            # During standard train / finetune, inference average the classifier
            # predictions.
            return (x + x_dist) / 2

        elif training:
            # Only return separate classification predictions when training in distilled
            # mode.
            return x, x_dist


"""
Let's verify if the `ViTDistilled` class can be initialized and called as expected.
"""

deit_tiny_distilled = ViTDistilled()

dummy_inputs = tf.ones((2, 224, 224, 3))
outputs = deit_tiny_distilled(dummy_inputs, training=False)
print(f"output_shape: {outputs.shape}")

"""
## Implementing the trainer

Unlike what happens in standard knowledge distillation
([Hinton et al.](https://arxiv.org/abs/1503.02531)),
where a temperature-scaled softmax is used as well as KL divergence,
DeiT authors use the following loss function:

![](https://i.imgur.com/bXdxsBq.png)


Here,

* CE is cross-entropy
* `psi` is the softmax function
* Z_s denotes student predictions
* y denotes true labels
* y_t denotes teacher predictions
"""


class DeiT(keras.Model):
    # Reference:
    # https://keras.io/examples/vision/knowledge_distillation/
    def __init__(self, student, teacher, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = keras.metrics.Mean(
            name="distillation_loss"
        )
        self.accuracy = keras.metrics.CategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [
            self.accuracy,
            self.student_loss_tracker,
            self.distillation_loss_tracker,
        ]

    def compile(
        self,
        optimizer,
        student_loss_fn,
        distillation_loss_fn,
        run_eagerly=False,
        jit_compile=False,
    ):
        super().compile(
            optimizer=optimizer,
            run_eagerly=run_eagerly,
            jit_compile=jit_compile,
        )
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn

    def train_step(self, data):
        # Unpack data.
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x)["dense"]
        teacher_predictions = tf.nn.softmax(teacher_predictions, axis=-1)

        with tf.GradientTape() as tape:
            # Forward pass of student.
            cls_predictions, dist_predictions = self.student(
                x / 255.0, training=True
            )

            # Compute losses.
            student_loss = self.student_loss_fn(y, cls_predictions)
            distillation_loss = self.distillation_loss_fn(
                teacher_predictions, dist_predictions
            )
            loss = (student_loss + distillation_loss) / 2

        # Compute gradients.
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights.
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        student_predictions = (cls_predictions + dist_predictions) / 2
        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)
        self.accuracy.update_state(y, student_predictions)

        # Return a dict of performance.
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data.
        x, y = data

        # Compute predictions.
        y_prediction = self.student(x / 255.0)

        # Calculate the loss.
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.student_loss_tracker.update_state(student_loss)
        self.accuracy.update_state(y, y_prediction)

        # Return a dict of performance.
        results = {m.name: m.result() for m in self.metrics}
        return results

    def call(self, inputs):
        return self.student(inputs / 255.0)


"""
## Load the teacher model

This model is based on the BiT family of ResNets
([Kolesnikov et al.](https://arxiv.org/abs/1912.11370))
fine-tuned on the `tf_flowers` dataset. You can refer to
[this notebook](https://github.com/sayakpaul/deit-tf/blob/main/notebooks/bit-teacher.ipynb)
to know how the training was performed. The teacher model has about 212 Million parameters
which is about **40x more** than the student.
"""

"""shell
wget -q https://github.com/sayakpaul/deit-tf/releases/download/v0.1.0/bit_teacher_flowers.zip
unzip -q bit_teacher_flowers.zip
"""

bit_teacher_flowers = keras.layers.TFSMLayer(
    filepath="bit_teacher_flowers",
    call_endpoint="serving_default",
)


"""
## Training through distillation
"""

deit_tiny = ViTDistilled()
deit_distiller = DeiT(student=deit_tiny, teacher=bit_teacher_flowers)

lr_scaled = (BASE_LR / 512) * BATCH_SIZE
deit_distiller.compile(
    optimizer=keras.optimizers.AdamW(
        weight_decay=WEIGHT_DECAY, learning_rate=lr_scaled
    ),
    student_loss_fn=keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1
    ),
    distillation_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
)
_ = deit_distiller.fit(
    train_dataset, validation_data=val_dataset, epochs=NUM_EPOCHS
)

"""
If we had trained the same model (the `ViTClassifier`) from scratch with the exact same
hyperparameters, the model would have scored about 59% accuracy. You can adapt the following code
to reproduce this result:

```
vit_tiny = ViTClassifier()

inputs = keras.Input((RESOLUTION, RESOLUTION, 3))
x = keras.layers.Rescaling(scale=1./255)(inputs)
outputs = deit_tiny(x)
model = keras.Model(inputs, outputs)

model.compile(...)
model.fit(...)
```
"""

"""
## Notes

* Through the use of distillation, we're effectively transferring the inductive biases of
a CNN-based teacher model.
* Interestingly enough, this distillation strategy works better with a CNN as the teacher
model rather than a Transformer as shown in the paper.
* The use of regularization to train DeiT models is very important.
* ViT models are initialized with a combination of different initializers including
truncated normal, random normal, Glorot uniform, etc. If you're looking for
end-to-end reproduction of the original results, don't forget to initialize the ViTs well.
* If you want to explore the pre-trained DeiT models in TensorFlow and Keras with code
for fine-tuning, [check out these models on TF-Hub](https://tfhub.dev/sayakpaul/collections/deit/1).

## Acknowledgements

* Ross Wightman for keeping
[`timm`](https://github.com/rwightman/pytorch-image-models)
updated with readable implementations. I referred to the implementations of ViT and DeiT
a lot during implementing them in TensorFlow.
* [Aritra Roy Gosthipaty](https://github.com/ariG23498)
who implemented some portions of the `ViTClassifier` in another project.
* [Google Developers Experts](https://developers.google.com/programs/experts/)
program for supporting me with GCP credits which were used to run experiments for this
example.
"""
