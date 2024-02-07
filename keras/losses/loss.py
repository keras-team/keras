import tree

from keras import backend
from keras import ops
from keras.api_export import keras_export
from keras.utils.naming import auto_name


@keras_export(["keras.Loss", "keras.losses.Loss"])
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

    def __init__(self, name=None, reduction="sum_over_batch_size", dtype=None):
        self.name = name or auto_name(self.__class__.__name__)
        self.reduction = standardize_reduction(reduction)
        self.dtype = dtype or backend.floatx()

    def __call__(self, y_true, y_pred, sample_weight=None):
        in_mask = getattr(y_pred, "_keras_mask", None)

        with ops.name_scope(self.name):
            y_pred = tree.map_structure(
                lambda x: ops.convert_to_tensor(x, dtype=self.dtype), y_pred
            )
            y_true = tree.map_structure(
                lambda x: ops.convert_to_tensor(x, dtype=self.dtype), y_true
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
                dtype=self.dtype,
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
            f"Expected one of {allowed}. Received: "
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


def reduce_weighted_values(
    values,
    sample_weight=None,
    mask=None,
    reduction="sum_over_batch_size",
    dtype=None,
):
    reduction = standardize_reduction(reduction)

    values = ops.convert_to_tensor(values, dtype=dtype)
    if sample_weight is not None:
        sample_weight = ops.convert_to_tensor(sample_weight, dtype=dtype)
        sample_weight, values = squeeze_to_same_rank(sample_weight, values)
        values = values * sample_weight

    values = apply_mask(values, mask)

    if reduction is None or reduction == "none":
        return values

    if reduction == "sum":
        return ops.sum(values)

    if reduction == "sum_over_batch_size":
        if mask is None:
            # batch_size is the total number of elements
            return ops.mean(values)
        batch_size = ops.count_nonzero(mask)
        values_sum = ops.sum(values)
        # safe divide
        return ops.cond(
            batch_size == 0,
            lambda: values_sum,  # will necessarily be all zeros
            lambda: values_sum / ops.cast(batch_size, dtype),
        )

    # we shouldn't get here because the call to `standardize_reduction`
    # at the top of this function should raise the exact error as below.
    allowed = {"sum_over_batch_size", "sum", None, "none"}
    raise ValueError(
        "Invalid value for argument `reduction`. "
        f"Expected one of {allowed}. Received: "
        f"reduction={reduction}"
    )


def apply_mask(values, mask):
    if mask is None:
        return values
    mask = ops.cast(mask, "bool")
    while len(mask.shape) < len(values.shape):
        mask = ops.expand_dims(mask, axis=-1)
    values, mask = squeeze_to_same_rank(values, mask)
    return ops.where(mask, values, ops.zeros_like(values))
