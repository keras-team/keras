import warnings

from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.metrics.metric import Metric
from keras.src.metrics.metrics_utils import confusion_matrix


class _IoUBase(Metric):
    """Computes the confusion matrix for Intersection-Over-Union metrics.

    Formula:

    ```python
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```
    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    From IoUs of individual classes, the MeanIoU can be computed as the mean of
    the individual IoUs.

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        num_classes: The possible number of labels the prediction task can have.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        ignore_class: Optional integer. The ID of a class to be ignored during
            metric computation. This is useful, for example, in segmentation
            problems featuring a "void" class (commonly -1 or 255) in
            segmentation maps. By default (`ignore_class=None`), all classes are
            considered.
        sparse_y_true: Whether labels are encoded using integers or
            dense floating point vectors. If `False`, the `argmax` function
            is used to determine each sample's most likely associated label.
        sparse_y_pred: Whether predictions are encoded using integers or
            dense floating point vectors. If `False`, the `argmax` function
            is used to determine each sample's most likely associated label.
        axis: (Optional) -1 is the dimension containing the logits.
            Defaults to `-1`.
    """

    def __init__(
        self,
        num_classes,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_true=True,
        sparse_y_pred=True,
        axis=-1,
    ):
        # defaulting to int to avoid issues with confusion matrix
        super().__init__(name=name, dtype=dtype or "int")
        # Metric should be maximized during optimization.
        self._direction = "up"
        self.num_classes = num_classes
        self.ignore_class = ignore_class
        self.sparse_y_true = sparse_y_true
        self.sparse_y_pred = sparse_y_pred
        self.axis = axis

        self.total_cm = self.add_variable(
            name="total_confusion_matrix",
            shape=(num_classes, num_classes),
            initializer=initializers.Zeros(),
            dtype=self.dtype,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        Args:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Can
                be a `Tensor` whose rank is either 0, or the same as `y_true`,
                and must be broadcastable to `y_true`. Defaults to `1`.

        Returns:
            Update op.
        """

        if not self.sparse_y_true:
            y_true = ops.argmax(y_true, axis=self.axis)
        if not self.sparse_y_pred:
            y_pred = ops.argmax(y_pred, axis=self.axis)

        y_true = ops.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = ops.convert_to_tensor(y_pred, dtype=self.dtype)

        # Flatten the input if its rank > 1.
        if len(y_pred.shape) > 1:
            y_pred = ops.reshape(y_pred, [-1])

        if len(y_true.shape) > 1:
            y_true = ops.reshape(y_true, [-1])

        if sample_weight is None:
            sample_weight = 1
        else:
            if (
                hasattr(sample_weight, "dtype")
                and "float" in str(sample_weight.dtype)
                and "int" in str(self.dtype)
            ):
                warnings.warn(
                    "You are passing weight as `float`, but dtype is `int`. "
                    "This may result in an incorrect weight due to type casting"
                    " Consider using integer weights."
                )
        sample_weight = ops.convert_to_tensor(sample_weight, dtype=self.dtype)

        if len(sample_weight.shape) > 1:
            sample_weight = ops.reshape(sample_weight, [-1])

        sample_weight = ops.broadcast_to(sample_weight, ops.shape(y_true))

        if self.ignore_class is not None:
            ignore_class = ops.convert_to_tensor(
                self.ignore_class, y_true.dtype
            )
            valid_mask = ops.not_equal(y_true, ignore_class)
            y_true = y_true * ops.cast(valid_mask, y_true.dtype)
            y_pred = y_pred * ops.cast(valid_mask, y_pred.dtype)
            if sample_weight is not None:
                sample_weight = sample_weight * ops.cast(
                    valid_mask, sample_weight.dtype
                )

        y_pred = ops.cast(y_pred, dtype=self.dtype)
        y_true = ops.cast(y_true, dtype=self.dtype)
        sample_weight = ops.cast(sample_weight, dtype=self.dtype)

        current_cm = confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=self.dtype,
        )

        return self.total_cm.assign(self.total_cm + current_cm)

    def reset_state(self):
        self.total_cm.assign(
            ops.zeros(self.total_cm.shape, dtype=self.total_cm.dtype)
        )


@keras_export("keras.metrics.IoU")
class IoU(_IoUBase):
    """Computes the Intersection-Over-Union metric for specific target classes.

    Formula:

    ```python
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```
    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Note, this class first computes IoUs for all individual classes, then
    returns the mean of IoUs for the classes that are specified by
    `target_class_ids`. If `target_class_ids` has only one id value, the IoU of
    that specific class is returned.

    Args:
        num_classes: The possible number of labels the prediction task can have.
        target_class_ids: A tuple or list of target class ids for which the
            metric is returned. To compute IoU for a specific class, a list
            (or tuple) of a single id value should be provided.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        ignore_class: Optional integer. The ID of a class to be ignored during
            metric computation. This is useful, for example, in segmentation
            problems featuring a "void" class (commonly -1 or 255) in
            segmentation maps. By default (`ignore_class=None`), all classes are
              considered.
        sparse_y_true: Whether labels are encoded using integers or
            dense floating point vectors. If `False`, the `argmax` function
            is used to determine each sample's most likely associated label.
        sparse_y_pred: Whether predictions are encoded using integers or
            dense floating point vectors. If `False`, the `argmax` function
            is used to determine each sample's most likely associated label.
        axis: (Optional) -1 is the dimension containing the logits.
            Defaults to `-1`.

    Examples:

    >>> # cm = [[1, 1],
    >>> #        [1, 1]]
    >>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
    >>> # iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # iou = [0.33, 0.33]
    >>> m = keras.metrics.IoU(num_classes=2, target_class_ids=[0])
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
    >>> m.result()
    0.33333334

    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
    ...                sample_weight=[0.3, 0.3, 0.3, 0.1])
    >>> # cm = [[0.3, 0.3],
    >>> #        [0.3, 0.1]]
    >>> # sum_row = [0.6, 0.4], sum_col = [0.6, 0.4],
    >>> # true_positives = [0.3, 0.1]
    >>> # iou = [0.33, 0.14]
    >>> m.result()
    0.33333334

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.IoU(num_classes=2, target_class_ids=[0])])
    ```
    """

    def __init__(
        self,
        num_classes,
        target_class_ids,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_true=True,
        sparse_y_pred=True,
        axis=-1,
    ):
        super().__init__(
            name=name,
            num_classes=num_classes,
            ignore_class=ignore_class,
            sparse_y_true=sparse_y_true,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
            dtype=dtype,
        )
        if max(target_class_ids) >= num_classes:
            raise ValueError(
                f"Target class id {max(target_class_ids)} "
                "is out of range, which is "
                f"[{0}, {num_classes})."
            )
        self.target_class_ids = list(target_class_ids)

    def result(self):
        """Compute the intersection-over-union via the confusion matrix."""
        sum_over_row = ops.cast(
            ops.sum(self.total_cm, axis=0), dtype=self.dtype
        )
        sum_over_col = ops.cast(
            ops.sum(self.total_cm, axis=1), dtype=self.dtype
        )
        true_positives = ops.cast(ops.diag(self.total_cm), dtype=self.dtype)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        target_class_ids = ops.convert_to_tensor(
            self.target_class_ids, dtype="int32"
        )

        # Only keep the target classes
        true_positives = ops.take_along_axis(
            true_positives, target_class_ids, axis=-1
        )
        denominator = ops.take_along_axis(
            denominator, target_class_ids, axis=-1
        )
        denominator = ops.cast(denominator, dtype="float32")

        # If the denominator is 0, we need to ignore the class.
        num_valid_entries = ops.sum(
            ops.cast(ops.greater(denominator, 1e-9), dtype="float32")
        )

        iou = ops.divide(true_positives, denominator + backend.epsilon())

        return ops.divide(
            ops.sum(iou, axis=self.axis), num_valid_entries + backend.epsilon()
        )

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "target_class_ids": self.target_class_ids,
            "ignore_class": self.ignore_class,
            "sparse_y_true": self.sparse_y_true,
            "sparse_y_pred": self.sparse_y_pred,
            "axis": self.axis,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export("keras.metrics.BinaryIoU")
class BinaryIoU(IoU):
    """Computes the Intersection-Over-Union metric for class 0 and/or 1.

    Formula:

    ```python
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```
    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    This class can be used to compute IoUs for a binary classification task
    where the predictions are provided as logits. First a `threshold` is applied
    to the predicted values such that those that are below the `threshold` are
    converted to class 0 and those that are above the `threshold` are converted
    to class 1.

    IoUs for classes 0 and 1 are then computed, the mean of IoUs for the classes
    that are specified by `target_class_ids` is returned.

    Note: with `threshold=0`, this metric has the same behavior as `IoU`.

    Args:
        target_class_ids: A tuple or list of target class ids for which the
            metric is returned. Options are `[0]`, `[1]`, or `[0, 1]`. With
            `[0]` (or `[1]`), the IoU metric for class 0 (or class 1,
            respectively) is returned. With `[0, 1]`, the mean of IoUs for the
            two classes is returned.
        threshold: A threshold that applies to the prediction logits to convert
            them to either predicted class 0 if the logit is below `threshold`
            or predicted class 1 if the logit is above `threshold`.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:

    >>> m = keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.3)
    >>> m.update_state([0, 1, 0, 1], [0.1, 0.2, 0.4, 0.7])
    >>> m.result()
    0.33333334

    >>> m.reset_state()
    >>> m.update_state([0, 1, 0, 1], [0.1, 0.2, 0.4, 0.7],
    ...                sample_weight=[0.2, 0.3, 0.4, 0.1])
    >>> # cm = [[0.2, 0.4],
    >>> #        [0.3, 0.1]]
    >>> # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5],
    >>> # true_positives = [0.2, 0.1]
    >>> # iou = [0.222, 0.125]
    >>> m.result()
    0.17361112

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.BinaryIoU(
            target_class_ids=[0],
            threshold=0.5
        )]
    )
    ```
    """

    def __init__(
        self,
        target_class_ids=(0, 1),
        threshold=0.5,
        name=None,
        dtype=None,
    ):
        super().__init__(
            num_classes=2,
            target_class_ids=target_class_ids,
            name=name,
            dtype=dtype,
        )
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        Before the confusion matrix is updated, the predicted values are
        thresholded to be:
            0 for values that are smaller than the `threshold`
            1 for values that are larger or equal to the `threshold`

        Args:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Can
                be a `Tensor` whose rank is either 0, or the same as `y_true`,
                and must be broadcastable to `y_true`. Defaults to `1`.

        Returns:
            Update op.
        """
        y_true = ops.convert_to_tensor(y_true, dtype=self.dtype)
        # convert y_pred on float 32 and cast just after to dtype
        y_pred = ops.convert_to_tensor(y_pred, dtype="float32")
        y_pred = ops.cast(y_pred >= self.threshold, self.dtype)
        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        return {
            "target_class_ids": self.target_class_ids,
            "threshold": self.threshold,
            "name": self.name,
            "dtype": self._dtype,
        }


@keras_export("keras.metrics.MeanIoU")
class MeanIoU(IoU):
    """Computes the mean Intersection-Over-Union metric.

    Formula:

    ```python
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```
    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Note that this class first computes IoUs for all individual classes, then
    returns the mean of these values.

    Args:
        num_classes: The possible number of labels the prediction task can have.
            This value must be provided, since a confusion matrix of dimension =
            [num_classes, num_classes] will be allocated.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        ignore_class: Optional integer. The ID of a class to be ignored during
            metric computation. This is useful, for example, in segmentation
            problems featuring a "void" class (commonly -1 or 255) in
            segmentation maps. By default (`ignore_class=None`), all classes are
            considered.
        sparse_y_true: Whether labels are encoded using integers or
            dense floating point vectors. If `False`, the `argmax` function
            is used to determine each sample's most likely associated label.
        sparse_y_pred: Whether predictions are encoded using integers or
            dense floating point vectors. If `False`, the `argmax` function
            is used to determine each sample's most likely associated label.
        axis: (Optional) The dimension containing the logits. Defaults to `-1`.


    Example:

    >>> # cm = [[1, 1],
    >>> #        [1, 1]]
    >>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
    >>> # iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2 = 0.33
    >>> m = keras.metrics.MeanIoU(num_classes=2)
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
    >>> m.result()
    0.33333334

    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
    ...                sample_weight=[0.3, 0.3, 0.3, 0.1])
    >>> m.result().numpy()
    0.23809525

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.MeanIoU(num_classes=2)])
    ```
    """

    def __init__(
        self,
        num_classes,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_true=True,
        sparse_y_pred=True,
        axis=-1,
    ):
        target_class_ids = list(range(num_classes))
        super().__init__(
            name=name,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            axis=axis,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=sparse_y_true,
            sparse_y_pred=sparse_y_pred,
        )

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "name": self.name,
            "dtype": self._dtype,
            "ignore_class": self.ignore_class,
            "sparse_y_true": self.sparse_y_true,
            "sparse_y_pred": self.sparse_y_pred,
            "axis": self.axis,
        }


@keras_export("keras.metrics.OneHotIoU")
class OneHotIoU(IoU):
    """Computes the Intersection-Over-Union metric for one-hot encoded labels.

    Formula:

    ```python
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```
    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    This class can be used to compute IoU for multi-class classification tasks
    where the labels are one-hot encoded (the last axis should have one
    dimension per class). Note that the predictions should also have the same
    shape. To compute the IoU, first the labels and predictions are converted
    back into integer format by taking the argmax over the class axis. Then the
    same computation steps as for the base `IoU` class apply.

    Note, if there is only one channel in the labels and predictions, this class
    is the same as class `IoU`. In this case, use `IoU` instead.

    Also, make sure that `num_classes` is equal to the number of classes in the
    data, to avoid a "labels out of bound" error when the confusion matrix is
    computed.

    Args:
        num_classes: The possible number of labels the prediction task can have.
        target_class_ids: A tuple or list of target class ids for which the
            metric is returned. To compute IoU for a specific class, a list
            (or tuple) of a single id value should be provided.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        ignore_class: Optional integer. The ID of a class to be ignored during
            metric computation. This is useful, for example, in segmentation
            problems featuring a "void" class (commonly -1 or 255) in
            segmentation maps. By default (`ignore_class=None`), all classes are
            considered.
        sparse_y_pred: Whether predictions are encoded using integers or
            dense floating point vectors. If `False`, the `argmax` function
            is used to determine each sample's most likely associated label.
        axis: (Optional) The dimension containing the logits. Defaults to `-1`.


    Example:

    >>> y_true = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> y_pred = np.array([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.5, 0.3, 0.1],
    ...                       [0.1, 0.4, 0.5]])
    >>> sample_weight = [0.1, 0.2, 0.3, 0.4]
    >>> m = keras.metrics.OneHotIoU(num_classes=3, target_class_ids=[0, 2])
    >>> m.update_state(
    ...     y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    >>> # cm = [[0, 0, 0.2+0.4],
    >>> #       [0.3, 0, 0],
    >>> #       [0, 0, 0.1]]
    >>> # sum_row = [0.3, 0, 0.7], sum_col = [0.6, 0.3, 0.1]
    >>> # true_positives = [0, 0, 0.1]
    >>> # single_iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # mean_iou = (0 / (0.3 + 0.6 - 0) + 0.1 / (0.7 + 0.1 - 0.1)) / 2
    >>> m.result()
    0.071

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.OneHotIoU(
            num_classes=3,
            target_class_id=[1]
        )]
    )
    ```
    """

    def __init__(
        self,
        num_classes,
        target_class_ids,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_pred=False,
        axis=-1,
    ):
        super().__init__(
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=False,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
        )

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "target_class_ids": self.target_class_ids,
            "name": self.name,
            "dtype": self._dtype,
            "ignore_class": self.ignore_class,
            "sparse_y_pred": self.sparse_y_pred,
            "axis": self.axis,
        }


@keras_export("keras.metrics.OneHotMeanIoU")
class OneHotMeanIoU(MeanIoU):
    """Computes mean Intersection-Over-Union metric for one-hot encoded labels.

    Formula:

    ```python
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```
    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    This class can be used to compute the mean IoU for multi-class
    classification tasks where the labels are one-hot encoded (the last axis
    should have one dimension per class). Note that the predictions should also
    have the same shape. To compute the mean IoU, first the labels and
    predictions are converted back into integer format by taking the argmax over
    the class axis. Then the same computation steps as for the base `MeanIoU`
    class apply.

    Note, if there is only one channel in the labels and predictions, this class
    is the same as class `MeanIoU`. In this case, use `MeanIoU` instead.

    Also, make sure that `num_classes` is equal to the number of classes in the
    data, to avoid a "labels out of bound" error when the confusion matrix is
    computed.

    Args:
        num_classes: The possible number of labels the prediction task can have.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        ignore_class: Optional integer. The ID of a class to be ignored during
            metric computation. This is useful, for example, in segmentation
            problems featuring a "void" class (commonly -1 or 255) in
            segmentation maps. By default (`ignore_class=None`), all classes are
            considered.
        sparse_y_pred: Whether predictions are encoded using natural numbers or
            probability distribution vectors. If `False`, the `argmax`
            function will be used to determine each sample's most likely
            associated label.
        axis: (Optional) The dimension containing the logits. Defaults to `-1`.


    Example:

    >>> y_true = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> y_pred = np.array([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.5, 0.3, 0.1],
    ...                       [0.1, 0.4, 0.5]])
    >>> sample_weight = [0.1, 0.2, 0.3, 0.4]
    >>> m = keras.metrics.OneHotMeanIoU(num_classes=3)
    >>> m.update_state(
    ...     y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    >>> # cm = [[0, 0, 0.2+0.4],
    >>> #       [0.3, 0, 0],
    >>> #       [0, 0, 0.1]]
    >>> # sum_row = [0.3, 0, 0.7], sum_col = [0.6, 0.3, 0.1]
    >>> # true_positives = [0, 0, 0.1]
    >>> # single_iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # mean_iou = (0 + 0 + 0.1 / (0.7 + 0.1 - 0.1)) / 3
    >>> m.result()
    0.048

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.OneHotMeanIoU(num_classes=3)])
    ```
    """

    def __init__(
        self,
        num_classes,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_pred=False,
        axis=-1,
    ):
        super().__init__(
            num_classes=num_classes,
            axis=axis,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=False,
            sparse_y_pred=sparse_y_pred,
        )

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "name": self.name,
            "dtype": self._dtype,
            "ignore_class": self.ignore_class,
            "sparse_y_pred": self.sparse_y_pred,
            "axis": self.axis,
        }
