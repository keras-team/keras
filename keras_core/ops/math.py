"""Commonly used math operations not included in NumPy."""

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.ops.operation import Operation
from keras_core.ops.operation_utils import reduce_shape


class SegmentSum(Operation):
    def __init__(self, num_segments=None, sorted=False):
        super().__init__()
        self.num_segments = num_segments
        self.sorted = sorted

    def compute_output_spec(self, data, segment_ids):
        num_segments = self.num_segments
        output_shape = (num_segments,) + tuple(data.shape[1:])
        return KerasTensor(shape=output_shape, dtype=data.dtype)

    def call(self, data, segment_ids):
        return backend.math.segment_sum(
            data,
            segment_ids,
            num_segments=self.num_segments,
            sorted=self.sorted,
        )


@keras_core_export("keras_core.ops.segment_sum")
def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    """Computes the sum of segments in a tensor.

    Args:
        data: Input tensor.
        segment_ids: A 1-D tensor containing segment indices for each
            element in `data`.
        num_segments: An integer representing the total number of
            segments. If not specified, it is inferred from the maximum
            value in `segment_ids`.
        sorted: A boolean indicating whether `segment_ids` is sorted.
            Default is `False`.

    Returns:
        A tensor containing the sum of segments, where each element
        represents the sum of the corresponding segment in `data`.

    Example:

    >>> data = keras_core.ops.convert_to_tensor([1, 2, 3, 4, 5, 6])
    >>> segment_ids = keras_core.ops.convert_to_tensor([0, 1, 0, 1, 0, 1])
    >>> segment_sum(data, segment_ids)
    array([9 12], shape=(2,), dtype=int32)
    """
    if any_symbolic_tensors((data,)):
        return SegmentSum(num_segments, sorted).symbolic_call(data, segment_ids)
    return backend.math.segment_sum(
        data, segment_ids, num_segments=num_segments, sorted=sorted
    )


class TopK(Operation):
    def __init__(self, k, sorted=False):
        super().__init__()
        self.k = k
        self.sorted = sorted

    def compute_output_spec(self, x):
        output_shape = list(x.shape)
        output_shape[-1] = self.k
        # Return a tuple (values, indices).
        return (
            KerasTensor(shape=output_shape, dtype=x.dtype),
            KerasTensor(shape=output_shape, dtype="int32"),
        )

    def call(self, x):
        return backend.math.top_k(x, self.k, self.sorted)


@keras_core_export("keras_core.ops.top_k")
def top_k(x, k, sorted=True):
    """Finds the top-k values and their indices in a tensor.

    Args:
        x: Input tensor.
        k: An integer representing the number of top elements to retrieve.
        sorted: A boolean indicating whether to sort the output in
        descending order. Default is `True`.

    Returns:
        A tuple containing two tensors. The first tensor contains the
        top-k values, and the second tensor contains the indices of the
        top-k values in the input tensor.

    Example:

    >>> x = keras_core.ops.convert_to_tensor([5, 2, 7, 1, 9, 3])
    >>> values, indices = top_k(x, k=3)
    >>> print(values)
    array([9 7 5], shape=(3,), dtype=int32)
    >>> print(indices)
    array([4 2 0], shape=(3,), dtype=int32)

    """
    if any_symbolic_tensors((x,)):
        return TopK(k, sorted).symbolic_call(x)
    return backend.math.top_k(x, k, sorted)


class InTopK(Operation):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def compute_output_spec(self, targets, predictions):
        return KerasTensor(shape=targets.shape, dtype="bool")

    def call(self, targets, predictions):
        return backend.math.in_top_k(targets, predictions, self.k)


@keras_core_export("keras_core.ops.in_top_k")
def in_top_k(targets, predictions, k):
    """Checks if the targets are in the top-k predictions.

    Args:
        targets: A tensor of true labels.
        predictions: A tensor of predicted labels.
        k: An integer representing the number of predictions to consider.

    Returns:
        A boolean tensor of the same shape as `targets`, where each element
        indicates whether the corresponding target is in the top-k predictions.

    Example:

    >>> targets = keras_core.ops.convert_to_tensor([2, 5, 3])
    >>> predictions = keras_core.ops.convert_to_tensor(
    ... [[0.1, 0.4, 0.6, 0.9, 0.5],
    ...  [0.1, 0.7, 0.9, 0.8, 0.3],
    ...  [0.1, 0.6, 0.9, 0.9, 0.5]])
    >>> in_top_k(targets, predictions, k=3)
    array([ True False  True], shape=(3,), dtype=bool)
    """
    if any_symbolic_tensors((targets, predictions)):
        return InTopK(k).symbolic_call(targets, predictions)
    return backend.math.in_top_k(targets, predictions, k)


class Logsumexp(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def compute_output_spec(self, x):
        output_shape = reduce_shape(x.shape, self.axis, self.keepdims)
        return KerasTensor(shape=output_shape)

    def call(self, x):
        return backend.math.logsumexp(x, axis=self.axis, keepdims=self.keepdims)


@keras_core_export("keras_core.ops.logsumexp")
def logsumexp(x, axis=None, keepdims=False):
    """Computes the logarithm of sum of exponentials of elements in a tensor.

    Args:
        x: Input tensor.
        axis: An integer or a tuple of integers specifying the axis/axes
            along which to compute the sum. If `None`, the sum is computed
            over all elements. Default is `None`.
        keepdims: A boolean indicating whether to keep the dimensions of
            the input tensor when computing the sum. Default is `False`.

    Returns:
        A tensor containing the logarithm of the sum of exponentials of
        elements in `x`.

    Example:

    >>> x = keras_core.ops.convert_to_tensor([1., 2., 3.])
    >>> logsumexp(x)
    3.407606
    """
    if any_symbolic_tensors((x,)):
        return Logsumexp(axis, keepdims).symbolic_call(x)
    return backend.math.logsumexp(x, axis=axis, keepdims=keepdims)


class Qr(Operation):
    def __init__(self, mode="reduced"):
        super().__init__()
        if mode not in {"reduced", "complete"}:
            raise ValueError(
                "`mode` argument value not supported. "
                "Expected one of {'reduced', 'complete'}. "
                f"Received: mode={mode}"
            )
        self.mode = mode

    def compute_output_spec(self, x):
        if len(x.shape) < 2:
            raise ValueError(
                "Input should have rank >= 2. Received: "
                f"input.shape = {x.shape}"
            )
        m = x.shape[-2]
        n = x.shape[-1]
        if m is None or n is None:
            raise ValueError(
                "Input should have its last 2 dimensions "
                "fully-defined. Received: "
                f"input.shape = {x.shape}"
            )
        k = min(m, n)
        base = tuple(x.shape[:-2])
        if self.mode == "reduced":
            return (
                KerasTensor(shape=base + (m, k), dtype=x.dtype),
                KerasTensor(shape=base + (k, n), dtype=x.dtype),
            )
        # 'complete' mode.
        return (
            KerasTensor(shape=base + (m, m), dtype=x.dtype),
            KerasTensor(shape=base + (m, n), dtype=x.dtype),
        )

    def call(self, x):
        return backend.math.qr(x, mode=self.mode)


@keras_core_export("keras_core.ops.qr")
def qr(x, mode="reduced"):
    """Computes the QR decomposition of a tensor.

    Args:
        x: Input tensor.
        mode: A string specifying the mode of the QR decomposition.
            - 'reduced': Returns the reduced QR decomposition. (default)
            - 'complete': Returns the complete QR decomposition.

    Returns:
        A tuple containing two tensors. The first tensor represents the
        orthogonal matrix Q, and the second tensor represents the upper
        triangular matrix R.

    Example:

    >>> x = keras_core.ops.convert_to_tensor([[1., 2.], [3., 4.], [5., 6.]])
    >>> q, r = qr(x)
    >>> print(q)
    array([[-0.16903079  0.897085]
           [-0.5070925   0.2760267 ]
           [-0.8451542  -0.34503305]], shape=(3, 2), dtype=float32)
    """

    if any_symbolic_tensors((x,)):
        return Qr(mode=mode).symbolic_call(x)
    return backend.math.qr(x, mode=mode)


class FFT(Operation):
    def compute_output_spec(self, a):
        if not isinstance(a, (tuple, list)) or len(a) != 2:
            raise ValueError(
                "Input `a` should be a tuple of two tensors - real and "
                f"imaginary. Received: a={a}"
            )

        real, imag = a
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `a` should be a tuple of two tensors - real and "
                "imaginary. Both the real and imaginary parts should have the "
                f"same shape. Received: a[0].shape = {real.shape}, "
                f"a[1].shape = {imag.shape}"
            )

        # We are calculating 1D FFT. Hence, rank >= 1.
        if len(real.shape) < 1:
            raise ValueError(
                f"Input should have rank >= 1. "
                f"Received: input.shape = {real.shape}"
            )

        # The axis along which we are calculating FFT should be fully-defined.
        m = real.shape[-1]
        if m is None:
            raise ValueError(
                f"Input should have its {self.axis}th axis fully-defined. "
                f"Received: input.shape = {real.shape}"
            )

        return (
            KerasTensor(shape=real.shape, dtype=real.dtype),
            KerasTensor(shape=imag.shape, dtype=imag.dtype),
        )

    def call(self, x):
        return backend.math.fft(x)


class FFT2(Operation):
    def compute_output_spec(self, a):
        if not isinstance(a, (tuple, list)) or len(a) != 2:
            raise ValueError(
                "Input `a` should be a tuple of two tensors - real and "
                f"imaginary. Received: a={a}"
            )

        real, imag = a
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `a` should be a tuple of two tensors - real and "
                "imaginary. Both the real and imaginary parts should have the "
                f"same shape. Received: a[0].shape = {real.shape}, "
                f"a[1].shape = {imag.shape}"
            )
        # We are calculating 2D FFT. Hence, rank >= 2.
        if len(real.shape) < 2:
            raise ValueError(
                f"Input should have rank >= 2. "
                f"Received: input.shape = {real.shape}"
            )

        # The axes along which we are calculating FFT should be fully-defined.
        m = real.shape[-1]
        n = real.shape[-2]
        if m is None or n is None:
            raise ValueError(
                f"Input should have its {self.axes} axes fully-defined. "
                f"Received: input.shape = {real.shape}"
            )

        return (
            KerasTensor(shape=real.shape, dtype=real.dtype),
            KerasTensor(shape=imag.shape, dtype=imag.dtype),
        )

    def call(self, x):
        return backend.math.fft2(x)


@keras_core_export("keras_core.ops.fft")
def fft(a):
    """Computes the Fast Fourier Transform along last axis of input.

    Args:
        a: Tuple of the real and imaginary parts of the input tensor. Both
            tensors in the tuple should be of floating type.

    Returns:
        A tuple containing two tensors - the real and imaginary parts of the
        output tensor.

    Example:

    >>> a = (
    ...     keras_core.ops.convert_to_tensor([1., 2.]),
    ...     keras_core.ops.convert_to_tensor([0., 1.]),
    ... )
    >>> fft(x)
    (array([ 3., -1.], dtype=float32), array([ 1., -1.], dtype=float32))
    """
    if any_symbolic_tensors(a):
        return FFT().symbolic_call(a)
    return backend.math.fft(a)


@keras_core_export("keras_core.ops.fft2")
def fft2(a):
    """Computes the 2D Fast Fourier Transform along the last two axes of input.

    Args:
        a: Tuple of the real and imaginary parts of the input tensor. Both
            tensors in the tuple should be of floating type.

    Returns:
        A tuple containing two tensors - the real and imaginary parts of the
        output.

    Example:

    >>> x = (
    ...     keras_core.ops.convert_to_tensor([[1., 2.], [2., 1.]]),
    ...     keras_core.ops.convert_to_tensor([[0., 1.], [1., 0.]]),
    ... )
    >>> fft2(x)
    (array([[ 6.,  0.],
        [ 0., -2.]], dtype=float32), array([[ 2.,  0.],
        [ 0., -2.]], dtype=float32))
    """
    if any_symbolic_tensors(a):
        return FFT2().symbolic_call(a)
    return backend.math.fft2(a)
