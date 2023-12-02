"""
Title: MobileViT: A mobile-friendly Transformer-based model for image classification
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2021/10/20
Last modified: 2023/11/23
Description: MobileViT for image classification with combined benefits of convolutions and Transformers.
Accelerator: GPU
Converted to Keras 3 by: [Pavan Kumar Singh](https://github.com/pksX01)
"""
"""
## Introduction

In this example, we implement the MobileViT architecture
([Mehta et al.](https://arxiv.org/abs/2110.02178)),
which combines the benefits of Transformers
([Vaswani et al.](https://arxiv.org/abs/1706.03762))
and convolutions. With Transformers, we can capture long-range dependencies that result
in global representations. With convolutions, we can capture spatial relationships that
model locality.

Besides combining the properties of Transformers and convolutions, the authors introduce
MobileViT as a general-purpose mobile-friendly backbone for different image recognition
tasks. Their findings suggest that, performance-wise, MobileViT is better than other
models with the same or higher complexity ([MobileNetV3](https://arxiv.org/abs/1905.02244),
for example), while being efficient on mobile devices.
"""

"""
## Imports
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf

from keras import layers
import keras as keras

import tensorflow_datasets as tfds

tfds.disable_progress_bar()

"""
## Hyperparameters
"""

# Values are from table 4.
patch_size = 4  # 2x2, for the Transformer blocks.
image_size = 256
expansion_factor = 2  # expansion factor for the MobileNetV2 blocks.

"""
## MobileViT utilities

The MobileViT architecture is comprised of the following blocks:

* Strided 3x3 convolutions that process the input image.
* [MobileNetV2](https://arxiv.org/abs/1801.04381)-style inverted residual blocks for
downsampling the resolution of the intermediate feature maps.
* MobileViT blocks that combine the benefits of Transformers and convolutions. It is
presented in the figure below (taken from the
[original paper](https://arxiv.org/abs/2110.02178)):


![](https://i.imgur.com/mANnhI7.png)
"""


def conv_block(x, filters=16, kernel_size=3, strides=2):
    conv_layer = layers.Conv2D(
        filters, kernel_size, strides=strides, activation=keras.activations.swish, padding="same"
    )
    return conv_layer(x)


# Reference: https://git.io/JKgtC


def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
    m = layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False)(x)
    m = layers.BatchNormalization()(m)
    m = keras.activations.swish(m)

    if strides == 2:
        m = layers.ZeroPadding2D()(m)
    m = layers.DepthwiseConv2D(
        3, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False
    )(m)
    m = layers.BatchNormalization()(m)
    m = keras.activations.swish(m)

    m = layers.Conv2D(output_channels, 1, padding="same", use_bias=False)(m)
    m = layers.BatchNormalization()(m)

    if x.shape[-1] == output_channels and strides == 1:
        return layers.Add()([m, x])
    return m


# Reference:
# https://keras.io/examples/vision/image_classification_with_vision_transformer/


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.swish)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_block(x, transformer_layers, projection_dim, num_heads=2):
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(
            x3,
            hidden_units=[x.shape[-1] * 2, x.shape[-1]],
            dropout_rate=0.1,
        )
        # Skip connection 2.
        x = layers.Add()([x3, x2])

    return x


def mobilevit_block(x, num_blocks, projection_dim, strides=1):
    # Local projection with convolutions.
    local_features = conv_block(x, filters=projection_dim, strides=strides)
    local_features = conv_block(
        local_features, filters=projection_dim, kernel_size=1, strides=strides
    )

    # Unfold into patches and then pass through Transformers.
    num_patches = int((local_features.shape[1] * local_features.shape[2]) / patch_size)
    non_overlapping_patches = layers.Reshape((patch_size, num_patches, projection_dim))(
        local_features
    )
    global_features = transformer_block(
        non_overlapping_patches, num_blocks, projection_dim
    )

    # Fold into conv-like feature-maps.
    folded_feature_map = layers.Reshape((*local_features.shape[1:-1], projection_dim))(
        global_features
    )

    # Apply point-wise conv -> concatenate with the input features.
    folded_feature_map = conv_block(
        folded_feature_map, filters=x.shape[-1], kernel_size=1, strides=strides
    )
    local_global_features = layers.Concatenate(axis=-1)([x, folded_feature_map])

    # Fuse the local and global features using a convoluion layer.
    local_global_features = conv_block(
        local_global_features, filters=projection_dim, strides=strides
    )

    return local_global_features


"""
**More on the MobileViT block**:

* First, the feature representations (A) go through convolution blocks that capture local
relationships. The expected shape of a single entry here would be `(h, w, num_channels)`.
* Then they get unfolded into another vector with shape `(p, n, num_channels)`,
where `p` is the area of a small patch, and `n` is `(h * w) / p`. So, we end up with `n`
non-overlapping patches.
* This unfolded vector is then passed through a Tranformer block that captures global
relationships between the patches.
* The output vector (B) is again folded into a vector of shape `(h, w, num_channels)`
resembling a feature map coming out of convolutions.

Vectors A and B are then passed through two more convolutional layers to fuse the local
and global representations. Notice how the spatial resolution of the final vector remains
unchanged at this point. The authors also present an explanation of how the MobileViT
block resembles a convolution block of a CNN. For more details, please refer to the
original paper.
"""

"""
Next, we combine these blocks together and implement the MobileViT architecture (XXS
variant). The following figure (taken from the original paper) presents a schematic
representation of the architecture:

![](https://i.ibb.co/sRbVRBN/image.png)
"""


def create_mobilevit(num_classes=5):
    inputs = keras.Input((image_size, image_size, 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Initial conv-stem -> MV2 block.
    x = conv_block(x, filters=16)
    x = inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=16
    )

    # Downsampling with MV2 block.
    x = inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=24, strides=2
    )
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=24
    )
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=24
    )

    # First MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=48, strides=2
    )
    x = mobilevit_block(x, num_blocks=2, projection_dim=64)

    # Second MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=64 * expansion_factor, output_channels=64, strides=2
    )
    x = mobilevit_block(x, num_blocks=4, projection_dim=80)

    # Third MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=80 * expansion_factor, output_channels=80, strides=2
    )
    x = mobilevit_block(x, num_blocks=3, projection_dim=96)
    x = conv_block(x, filters=320, kernel_size=1, strides=1)

    # Classification head.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


mobilevit_xxs = create_mobilevit()
mobilevit_xxs.summary()

"""
## Dataset preparation

We will be using the
[`tf_flowers`](https://www.tensorflow.org/datasets/catalog/tf_flowers)
dataset to demonstrate the model. Unlike other Transformer-based architectures,
MobileViT uses a simple augmentation pipeline primarily because it has the properties
of a CNN.
"""

batch_size = 64
auto = tf.data.AUTOTUNE
resize_bigger = 280
num_classes = 5


def preprocess_dataset(is_training=True):
    def _pp(image, label):
        if is_training:
            # Resize to a bigger spatial resolution and take the random
            # crops.
            image = tf.image.resize(image, (resize_bigger, resize_bigger))
            image = tf.image.random_crop(image, (image_size, image_size, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, (image_size, image_size))
        label = tf.one_hot(label, depth=num_classes)
        return image, label

    return _pp


def prepare_dataset(dataset, is_training=True):
    if is_training:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(preprocess_dataset(is_training), num_parallel_calls=auto)
    return dataset.batch(batch_size).prefetch(auto)


"""
The authors use a multi-scale data sampler to help the model learn representations of
varied scales. In this example, we discard this part.
"""

"""
## Load and prepare the dataset
"""

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
## Train a MobileViT (XXS) model
"""

learning_rate = 0.002
label_smoothing_factor = 0.1
epochs = 30

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_factor)


def run_experiment(epochs=epochs):
    mobilevit_xxs = create_mobilevit(num_classes=num_classes)
    mobilevit_xxs.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    checkpoint_filepath = "/tmp/checkpoint.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    mobilevit_xxs.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback],
    )
    mobilevit_xxs.load_weights(checkpoint_filepath)
    _, accuracy = mobilevit_xxs.evaluate(val_dataset)
    print(f"Validation accuracy: {round(accuracy * 100, 2)}%")
    return mobilevit_xxs


mobilevit_xxs = run_experiment()

"""
## Results and TFLite conversion

With about one million parameters, getting to ~85% top-1 accuracy on 256x256 resolution is
a strong result. This MobileViT mobile is fully compatible with TensorFlow Lite (TFLite)
and can be converted with the following code:
"""

# Serialize the model as a SavedModel.
tf.saved_model.save(mobilevit_xxs, "mobilevit_xxs")

# Convert to TFLite. This form of quantization is called
# post-training dynamic-range quantization in TFLite.
converter = tf.lite.TFLiteConverter.from_saved_model("mobilevit_xxs")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS,  # Enable TensorFlow ops.
]
tflite_model = converter.convert()
open("mobilevit_xxs.tflite", "wb").write(tflite_model)

"""
To learn more about different quantization recipes available in TFLite and running
inference with TFLite models, check out
[this official resource](https://www.tensorflow.org/lite/performance/post_training_quantization).
"""