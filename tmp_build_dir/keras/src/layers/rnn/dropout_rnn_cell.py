from keras.src import backend
from keras.src import ops


class DropoutRNNCell:
    """Object that holds dropout-related functionality for RNN cells.

    This class is not a standalone RNN cell. It suppose to be used with a RNN
    cell by multiple inheritance. Any cell that mix with class should have
    following fields:

    - `dropout`: a float number in the range `[0, 1]`.
        Dropout rate for the input tensor.
    - `recurrent_dropout`: a float number in the range `[0, 1]`.
        Dropout rate for the recurrent connections.
    - `seed_generator`, an instance of `backend.random.SeedGenerator`.

    This object will create and cache dropout masks, and reuse them for
    all incoming steps, so that the same mask is used for every step.
    """

    def _create_dropout_mask(self, step_input, dropout_rate):
        count = getattr(self, "dropout_mask_count", None)
        ones = ops.ones_like(step_input)
        if count is None:
            return backend.random.dropout(
                ones, rate=dropout_rate, seed=self.seed_generator
            )
        else:
            return [
                backend.random.dropout(
                    ones, rate=dropout_rate, seed=self.seed_generator
                )
                for _ in range(count)
            ]

    def get_dropout_mask(self, step_input):
        if not hasattr(self, "_dropout_mask"):
            self._dropout_mask = None
        if self._dropout_mask is None and self.dropout > 0:
            self._dropout_mask = self._create_dropout_mask(
                step_input, self.dropout
            )
        return self._dropout_mask

    def get_recurrent_dropout_mask(self, step_input):
        if not hasattr(self, "_recurrent_dropout_mask"):
            self._recurrent_dropout_mask = None
        if self._recurrent_dropout_mask is None and self.recurrent_dropout > 0:
            self._recurrent_dropout_mask = self._create_dropout_mask(
                step_input, self.recurrent_dropout
            )
        return self._recurrent_dropout_mask

    def reset_dropout_mask(self):
        """Reset the cached dropout mask if any.

        The RNN layer invokes this in the `call()` method
        so that the cached mask is cleared after calling `cell.call()`. The
        mask should be cached across all timestep within the same batch, but
        shouldn't be cached between batches.
        """
        self._dropout_mask = None

    def reset_recurrent_dropout_mask(self):
        self._recurrent_dropout_mask = None
