from tensorflow import nest

from keras_core import backend
from keras_core import operations as ops
from keras_core.api_export import keras_core_export
from keras_core.utils import dtype_utils
from keras_core.utils.naming import auto_name


@keras_core_export(["keras_core.Loss", "keras_core.losses.Loss"])
class Loss:
    """Loss base class.

    To be implemented by subclasses:

    * `call()`: Contains the logic for loss calculation using `y_true`,
        `y_pred`.

    Example subclass implementation:

    ```python
    class MeanSquaredError(Loss):
        def call(self, y_true, y_pred):
            return ops.mean(ops.square(y_pred - y_true), axis=-1)
    ```
    """

    def __init__(self, name=None, reduction="sum_over_batch_size"):
        self.name = name or auto_name(self.__class__.__name__)
        self.reduction = standardize_reduction(reduction)

    def __call__(self, y_true, y_pred, sample_weight=None):
        in_mask = getattr(y_pred, "_keras_mask", None)

        with ops.name_scope(self.name):
            dtype = backend.floatx()
            y_pred = nest.map_structure(
                lambda x: ops.convert_to_tensor(x, dtype=dtype), y_pred
            )
            y_true = nest.map_structure(
                lambda x: ops.convert_to_tensor(x, dtype=dtype), y_true
            )

            losses = self.call(y_true, y_pred)
            out_mask = getattr(losses, "_keras_mask", None)

            if in_mask is not None and out_mask is not None:
                mask = in_mask & out_mask
            elif in_mask is not None:
                mask = in_mask
            elif out_mask is not None:
                mask = out_mask
            else:
                mask = None

            return reduce_weighted_values(
                losses,
                sample_weight=sample_weight,
                mask=mask,
                reduction=self.reduction,
            )

    def call(self, y_true, y_pred):
        raise NotImplementedError

    def get_config(self):
        return {"name": self.name, "reduction": self.reduction}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def standardize_reduction(reduction):
    allowed = {"sum_over_batch_size", "sum", None, "none"}
    if reduction not in allowed:
        raise ValueError(
            "Invalid value for argument `reduction`. "
            f"Expected on of {allowed}. Received: "
            f"reduction={reduction}"
        )
    return reduction


def squeeze_to_same_rank(x1, x2):
    """Squeeze last dim if ranks differ from expected by exactly 1."""
    x1_rank = len(x1.shape)
    x2_rank = len(x2.shape)
    if x1_rank == x2_rank:
        return x1, x2
    if x1_rank == x2_rank + 1:
        if x1.shape[-1] == 1:
            x1 = ops.squeeze(x1, axis=-1)
    if x2_rank == x1_rank + 1:
        if x2.shape[-1] == 1:
            x2 = ops.squeeze(x2, axis=-1)
    return x1, x2


def reduce_values(values, reduction="sum_over_batch_size"):
    if (
        reduction is None
        or reduction == "none"
        or tuple(values.shape) == ()
        or tuple(values.shape) == (0,)
    ):
        return values
    loss = ops.sum(values)
    if reduction == "sum_over_batch_size":
        loss /= ops.cast(
            ops.prod(ops.convert_to_tensor(ops.shape(values), dtype="int32")),
            loss.dtype,
        )
    return loss


def reduce_weighted_values(
    values,
    sample_weight=None,
    mask=None,
    reduction="sum_over_batch_size",
):
    reduction = standardize_reduction(reduction)

    values = ops.convert_to_tensor(values)
    if sample_weight is not None:
        sample_weight = ops.convert_to_tensor(sample_weight, dtype=values.dtype)
    if mask is not None:
        mask = ops.convert_to_tensor(mask, dtype=values.dtype)

    # Merge mask and sample weight into sample weight.
    sample_weight = apply_mask(
        sample_weight, mask, dtype=values.dtype, reduction=reduction
    )

    # Convert any non float dtypes to floats, to avoid loss of precision
    # for dtype like int or bool.
    dtype = backend.standardize_dtype(values.dtype)
    if not dtype_utils.is_float(dtype):
        input_dtype = values.dtype
        values = ops.cast(values, "float32")
        input_casted = True
    else:
        input_casted = False

    if sample_weight is not None:
        sample_weight = ops.cast(sample_weight, values.dtype)
        # Update dimensions of `sample_weight` to match `losses`.
        values, sample_weight = squeeze_to_same_rank(values, sample_weight)
        values = values * sample_weight

    # Apply reduction function to the individual weighted losses.
    loss = reduce_values(values, reduction)

    if input_casted:
        # Convert the result back to the input type.
        loss = ops.cast(loss, input_dtype)
    return loss


def apply_mask(sample_weight, mask, dtype, reduction):
    """Applies any mask on predictions to sample weights."""
    if mask is not None:
        mask = ops.cast(mask, dtype=dtype)
        if reduction == "sum_over_batch_size":
            # Valid entries have weight `total/valid`, while invalid ones
            # have 0. When summed over batch, they will be reduced to:
            #
            # mean(loss * sample_weight * total / valid)
            #   = sum(loss * sample_weight * total / valid) / total
            #   = sum(loss * sample_weight) / total * total / valid
            #   = sum(loss * sample_weight) / valid
            total = ops.cast(ops.shape(mask)[0], dtype=dtype)
            valid = ops.sum(mask)  # May be 0!
            mask *= total / (valid + backend.epsilon())

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, dtype=dtype)
            mask, sample_weight = squeeze_to_same_rank(mask, sample_weight)
            sample_weight *= mask
        else:
            sample_weight = mask
    return sample_weight
