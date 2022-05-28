"""The DetermisticRandomTestTool.

(from www.tensorflow.org/guide/migrate/validate_correctness) is a tool used to
make random number generation semantics match between TF1.x graphs/sessions and
eager execution.
"""

import sys

import tensorflow.compat.v2 as tf

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export(v1=["keras.utils.DeterministicRandomTestTool"])
class DeterministicRandomTestTool(object):
    """DeterministicRandomTestTool is a testing tool.

    This tool is used to validate random number generation semantics match
    between TF1.x graphs/sessions and eager execution.

    This is useful when you are migrating from TF 1.x to TF2 and need to make
    sure your computation is still happening correctly along the way. See the
    validating correctness migration guide for more info:
    https://www.tensorflow.org/guide/migrate/validate_correctness

    The following DeterministicRandomTestTool object provides a context manager
    scope() that can make stateful random operations use the same seed across
    both TF1 graphs/sessions and eager execution,The tool provides two testing
    modes:
    - constant which uses the same seed for every single operation no matter how
    many times it has been called and,
    - num_random_ops which uses the number of previously-observed stateful
    random operations as the operation seed.
    The num_random_ops mode serves as a more sensitive validation check than the
    constant mode. It ensures that the random numbers initialization does not
    get accidentaly reused.(for example if several weights take on the same
    initializations), you can use the num_random_ops mode to avoid this. In the
    num_random_ops mode, the generated random numbers will depend on the
    ordering of random ops in the program.

    This applies both to the stateful random operations used for creating and
    initializing variables, and to the stateful random operations used in
    computation (such as for dropout layers).
    """

    def __init__(self, seed: int = 42, mode="constant"):
        """Set mode to 'constant' or 'num_random_ops'. Defaults to
        'constant'."""
        if mode not in {"constant", "num_random_ops"}:
            raise ValueError(
                "Mode arg must be 'constant' or 'num_random_ops'. "
                + "Got: {}".format(mode)
            )
        self.seed_implementation = sys.modules[tf.compat.v1.get_seed.__module__]
        self._mode = mode
        self._seed = seed
        self.operation_seed = 0
        self._observed_seeds = set()

    @property
    def operation_seed(self):
        return self._operation_seed

    @operation_seed.setter
    def operation_seed(self, value):
        self._operation_seed = value

    def scope(self):
        """set random seed."""

        tf.random.set_seed(self._seed)

        def _get_seed(_):
            """Wraps TF get_seed to make deterministic random generation easier.

            This makes a variable's initialization (and calls that involve
            random number generation) depend only on how many random number
            generations were used in the scope so far, rather than on how many
            unrelated operations the graph contains.

            Returns:
              Random seed tuple.
            """
            op_seed = self._operation_seed
            if self._mode == "constant":
                tf.random.set_seed(op_seed)
            else:
                if op_seed in self._observed_seeds:
                    raise ValueError(
                        "This `DeterministicRandomTestTool` "
                        "object is trying to re-use the "
                        + "already-used operation seed {}. ".format(op_seed)
                        + "It cannot guarantee random numbers will match "
                        + "between eager and sessions when an operation seed "
                        + "is reused. You most likely set "
                        + "`operation_seed` explicitly but used a value that "
                        + "caused the naturally-incrementing operation seed "
                        + "sequences to overlap with an already-used seed."
                    )

                self._observed_seeds.add(op_seed)
                self._operation_seed += 1

            return (self._seed, op_seed)

        # mock.patch internal symbols to modify the behavior of TF APIs relying
        # on them

        return tf.compat.v1.test.mock.patch.object(
            self.seed_implementation, "get_seed", wraps=_get_seed
        )
