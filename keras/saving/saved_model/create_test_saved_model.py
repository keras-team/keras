"""A binary that creates a serialized SavedModel from a keras model.

This is used in tests to ensure that model serialization is deterministic across
different processes.
"""

from absl import app
from absl import flags
from keras import regularizers
from keras import testing_utils

import tensorflow.compat.v2 as tf

flags.DEFINE_string('output_path', '', 'The path to write the SavedModel at.')

FLAGS = flags.FLAGS


def main(_) -> None:
  with testing_utils.model_type_scope('functional'):
    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    model.layers[-1].activity_regularizer = regularizers.get('l2')
    model.activity_regularizer = regularizers.get('l2')
    model.compile(
        loss='mse',
        optimizer='rmsprop')
    def callable_loss():
      return tf.reduce_sum(model.weights[0])
    model.add_loss(callable_loss)

    print(f'_____Writing saved model to: {FLAGS.output_path}')
    model.save(FLAGS.output_path)


if __name__ == '__main__':
  app.run(main)
