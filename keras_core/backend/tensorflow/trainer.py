import tensorflow as tf

from keras_core.trainers import trainer


class Trainer(trainer.Trainer):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.loss(y, y_pred)

        # Compute gradients
        trainable_weights = self.trainable_weights
        gradients = tape.gradient(loss, trainable_weights)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))
        return loss
