import tensorflow as tf
from tensorflow import keras

from keras.integration_test.models.input_spec import InputSpec
from keras.saving import serialization_lib

IMG_SIZE = (64, 64)
LATENT_DIM = 128


def get_data_spec(batch_size):
    return InputSpec((batch_size,) + IMG_SIZE + (3,))


def get_input_preprocessor():
    return None


class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn, jit_compile=False):
        super(GAN, self).compile(jit_compile=jit_compile)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )
        generated_images = self.generator(random_latent_vectors)
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )
        misleading_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(
                self.generator(random_latent_vectors)
            )
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights)
        )
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }

    def get_config(self):
        return {
            "discriminator": self.discriminator,
            "generator": self.generator,
            "latent_dim": self.latent_dim,
        }

    @classmethod
    def from_config(cls, config):
        discriminator = serialization_lib.deserialize_keras_object(
            config["discriminator"]
        )
        generator = serialization_lib.deserialize_keras_object(
            config["generator"]
        )
        latent_dim = config["latent_dim"]
        return cls(discriminator, generator, latent_dim)

    def get_compile_config(self):
        return {
            "loss_fn": self.loss_fn,
            "d_optimizer": self.d_optimizer,
            "g_optimizer": self.g_optimizer,
            "jit_compile": self.jit_compile,
        }

    def compile_from_config(self, config):
        loss_fn = serialization_lib.deserialize_keras_object(config["loss_fn"])
        d_optimizer = serialization_lib.deserialize_keras_object(
            config["d_optimizer"]
        )
        g_optimizer = serialization_lib.deserialize_keras_object(
            config["g_optimizer"]
        )
        jit_compile = config["jit_compile"]
        self.compile(
            loss_fn=loss_fn,
            d_optimizer=d_optimizer,
            g_optimizer=g_optimizer,
            jit_compile=jit_compile,
        )


def get_model(
    build=False, compile=False, jit_compile=False, include_preprocessing=True
):
    discriminator = keras.Sequential(
        [
            keras.Input(shape=IMG_SIZE + (3,)),
            keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )

    generator = keras.Sequential(
        [
            keras.Input(shape=(LATENT_DIM,)),
            keras.layers.Dense(8 * 8 * 128),
            keras.layers.Reshape((8, 8, 128)),
            keras.layers.Conv2DTranspose(
                128, kernel_size=4, strides=2, padding="same"
            ),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2DTranspose(
                256, kernel_size=4, strides=2, padding="same"
            ),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2DTranspose(
                512, kernel_size=4, strides=2, padding="same"
            ),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2D(
                3, kernel_size=5, padding="same", activation="sigmoid"
            ),
        ],
        name="generator",
    )

    gan = GAN(
        discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM
    )
    if compile:
        gan.compile(
            d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss_fn=keras.losses.BinaryCrossentropy(),
            jit_compile=jit_compile,
        )
    return gan


def get_custom_objects():
    return {"GAN": GAN}
