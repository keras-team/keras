"""Variable autoencoder.

Adapted from https://keras.io/examples/generative/vae/
"""

import tensorflow as tf
from tensorflow import keras

from keras.integration_test.models.input_spec import InputSpec
from keras.saving import serialization_lib

IMG_SIZE = (28, 28)
LATENT_DIM = 64


def get_input_preprocessor():
    return None


class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def get_config(self):
        base_config = super().get_config()
        return {
            "encoder": self.encoder,
            "decoder": self.decoder,
            **base_config,
        }

    @classmethod
    def from_config(cls, config):
        encoder = serialization_lib.deserialize_keras_object(
            config.pop("encoder")
        )
        decoder = serialization_lib.deserialize_keras_object(
            config.pop("decoder")
        )
        return cls(encoder, decoder, **config)


def get_data_spec(batch_size):
    return InputSpec((batch_size,) + IMG_SIZE + (1,))


def get_model(
    build=False, compile=False, jit_compile=False, include_preprocessing=True
):
    encoder_inputs = keras.Input(shape=IMG_SIZE + (1,))
    x = keras.layers.Conv2D(
        32, 3, activation="relu", strides=2, padding="same"
    )(encoder_inputs)
    x = keras.layers.Conv2D(
        64, 3, activation="relu", strides=2, padding="same"
    )(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    z_mean = keras.layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = keras.layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(
        encoder_inputs, [z_mean, z_log_var, z], name="encoder"
    )

    latent_inputs = keras.Input(shape=(LATENT_DIM,))
    x = keras.layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = keras.layers.Reshape((7, 7, 64))(x)
    x = keras.layers.Conv2DTranspose(
        64, 3, activation="relu", strides=2, padding="same"
    )(x)
    x = keras.layers.Conv2DTranspose(
        32, 3, activation="relu", strides=2, padding="same"
    )(x)
    decoder_outputs = keras.layers.Conv2DTranspose(
        1, 3, activation="sigmoid", padding="same"
    )(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    vae = VAE(encoder, decoder)
    if compile:
        vae.compile(optimizer=keras.optimizers.Adam(), jit_compile=jit_compile)
    return vae


def get_custom_objects():
    return {"VAE": VAE, "Sampling": Sampling}
