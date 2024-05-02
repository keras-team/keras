"""Commonly used math operations not included in NumPy."""

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape


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


@keras_export("keras.ops.segment_sum")
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
            Defaults to`False`.

    Returns:
        A tensor containing the sum of segments, where each element
        represents the sum of the corresponding segment in `data`.

    Example:

    >>> data = keras.ops.convert_to_tensor([1, 2, 10, 20, 100, 200])
    >>> segment_ids = keras.ops.convert_to_tensor([0, 0, 1, 1, 2, 2])
    >>> num_segments = 3
    >>> keras.ops.segment_sum(data, segment_ids,num_segments)
    array([3, 30, 300], dtype=int32)
    """
    if any_symbolic_tensors((data,)):
        return SegmentSum(num_segments, sorted).symbolic_call(data, segment_ids)
    return backend.math.segment_sum(
        data, segment_ids, num_segments=num_segments, sorted=sorted
    )


class SegmentMax(Operation):
    def __init__(self, num_segments=None, sorted=False):
        super().__init__()
        self.num_segments = num_segments
        self.sorted = sorted

    def compute_output_spec(self, data, segment_ids):
        num_segments = self.num_segments
        output_shape = (num_segments,) + tuple(data.shape[1:])
        return KerasTensor(shape=output_shape, dtype=data.dtype)

    def call(self, data, segment_ids):
        return backend.math.segment_max(
            data,
            segment_ids,
            num_segments=self.num_segments,
            sorted=self.sorted,
        )


@keras_export("keras.ops.segment_max")
def segment_max(data, segment_ids, num_segments=None, sorted=False):
    """Computes the max of segments in a tensor.

    Args:
        data: Input tensor.
        segment_ids: A 1-D tensor containing segment indices for each
            element in `data`.
        num_segments: An integer representing the total number of
            segments. If not specified, it is inferred from the maximum
            value in `segment_ids`.
        sorted: A boolean indicating whether `segment_ids` is sorted.
            Defaults to`False`.

    Returns:
        A tensor containing the max of segments, where each element
        represents the max of the corresponding segment in `data`.

    Example:

    >>> data = keras.ops.convert_to_tensor([1, 2, 10, 20, 100, 200])
    >>> segment_ids = keras.ops.convert_to_tensor([0, 0, 1, 1, 2, 2])
    >>> num_segments = 3
    >>> keras.ops.segment_max(data, segment_ids, num_segments)
    array([2, 20, 200], dtype=int32)
    """
    if any_symbolic_tensors((data,)):
        return SegmentMax(num_segments, sorted).symbolic_call(data, segment_ids)
    return backend.math.segment_max(
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


@keras_export("keras.ops.top_k")
def top_k(x, k, sorted=True):
    """Finds the top-k values and their indices in a tensor.

    Args:
        x: Input tensor.
        k: An integer representing the number of top elements to retrieve.
        sorted: A boolean indicating whether to sort the output in
        descending order. Defaults to`True`.

    Returns:
        A tuple containing two tensors. The first tensor contains the
        top-k values, and the second tensor contains the indices of the
        top-k values in the input tensor.

    Example:

    >>> x = keras.ops.convert_to_tensor([5, 2, 7, 1, 9, 3])
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


@keras_export("keras.ops.in_top_k")
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

    >>> targets = keras.ops.convert_to_tensor([2, 5, 3])
    >>> predictions = keras.ops.convert_to_tensor(
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


@keras_export("keras.ops.logsumexp")
def logsumexp(x, axis=None, keepdims=False):
    """Computes the logarithm of sum of exponentials of elements in a tensor.

    Args:
        x: Input tensor.
        axis: An integer or a tuple of integers specifying the axis/axes
            along which to compute the sum. If `None`, the sum is computed
            over all elements. Defaults to`None`.
        keepdims: A boolean indicating whether to keep the dimensions of
            the input tensor when computing the sum. Defaults to`False`.

    Returns:
        A tensor containing the logarithm of the sum of exponentials of
        elements in `x`.

    Example:

    >>> x = keras.ops.convert_to_tensor([1., 2., 3.])
    >>> logsumexp(x)
    3.407606
    """
    if any_symbolic_tensors((x,)):
        return Logsumexp(axis, keepdims).symbolic_call(x)
    return backend.math.logsumexp(x, axis=axis, keepdims=keepdims)


class ExtractSequences(Operation):
    def __init__(self, sequence_length, sequence_stride):
        super().__init__()
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride

    def compute_output_spec(self, x):
        if len(x.shape) < 1:
            raise ValueError(
                f"Input should have rank >= 1. "
                f"Received: input.shape = {x.shape}"
            )
        if x.shape[-1] is not None:
            num_sequences = (
                1 + (x.shape[-1] - self.sequence_length) // self.sequence_stride
            )
        else:
            num_sequences = None
        new_shape = x.shape[:-1] + (num_sequences, self.sequence_length)
        return KerasTensor(shape=new_shape, dtype=x.dtype)

    def call(self, x):
        return backend.math.extract_sequences(
            x,
            sequence_length=self.sequence_length,
            sequence_stride=self.sequence_stride,
        )


@keras_export("keras.ops.extract_sequences")
def extract_sequences(x, sequence_length, sequence_stride):
    """Expands the dimension of last axis into sequences of `sequence_length`.

    Slides a window of size `sequence_length` over the last axis of the input
    with a stride of `sequence_stride`, replacing the last axis with
    `[num_sequences, sequence_length]` sequences.

    If the dimension along the last axis is N, the number of sequences can be
    computed by:

    `num_sequences = 1 + (N - sequence_length) // sequence_stride`

    Args:
        x: Input tensor.
        sequence_length: An integer representing the sequences length.
        sequence_stride: An integer representing the sequences hop size.

    Returns:
        A tensor of sequences with shape [..., num_sequences, sequence_length].

    Example:

    >>> x = keras.ops.convert_to_tensor([1, 2, 3, 4, 5, 6])
    >>> extract_sequences(x, 3, 2)
    array([[1, 2, 3],
       [3, 4, 5]])
    """
    if any_symbolic_tensors((x,)):
        return ExtractSequences(sequence_length, sequence_stride).symbolic_call(
            x
        )
    return backend.math.extract_sequences(x, sequence_length, sequence_stride)


class FFT(Operation):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def compute_output_spec(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                f"imaginary. Received: x={x}"
            )

        real, imag = x
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                "imaginary. Both the real and imaginary parts should have the "
                f"same shape. Received: x[0].shape = {real.shape}, "
                f"x[1].shape = {imag.shape}"
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


@keras_export("keras.ops.fft")
def fft(x):
    """Computes the Fast Fourier Transform along last axis of input.

    Args:
        x: Tuple of the real and imaginary parts of the input tensor. Both
            tensors in the tuple should be of floating type.

    Returns:
        A tuple containing two tensors - the real and imaginary parts of the
        output tensor.

    Example:

    >>> x = (
    ...     keras.ops.convert_to_tensor([1., 2.]),
    ...     keras.ops.convert_to_tensor([0., 1.]),
    ... )
    >>> fft(x)
    (array([ 3., -1.], dtype=float32), array([ 1., -1.], dtype=float32))
    """
    if any_symbolic_tensors(x):
        return FFT().symbolic_call(x)
    return backend.math.fft(x)


class FFT2(Operation):
    def __init__(self):
        super().__init__()
        self.axes = (-2, -1)

    def compute_output_spec(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                f"imaginary. Received: x={x}"
            )

        real, imag = x
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                "imaginary. Both the real and imaginary parts should have the "
                f"same shape. Received: x[0].shape = {real.shape}, "
                f"x[1].shape = {imag.shape}"
            )
        # We are calculating 2D FFT. Hence, rank >= 2.
        if len(real.shape) < 2:
            raise ValueError(
                f"Input should have rank >= 2. "
                f"Received: input.shape = {real.shape}"
            )

        # The axes along which we are calculating FFT should be fully-defined.
        m = real.shape[self.axes[0]]
        n = real.shape[self.axes[1]]
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


@keras_export("keras.ops.fft2")
def fft2(x):
    """Computes the 2D Fast Fourier Transform along the last two axes of input.

    Args:
        x: Tuple of the real and imaginary parts of the input tensor. Both
            tensors in the tuple should be of floating type.

    Returns:
        A tuple containing two tensors - the real and imaginary parts of the
        output.

    Example:

    >>> x = (
    ...     keras.ops.convert_to_tensor([[1., 2.], [2., 1.]]),
    ...     keras.ops.convert_to_tensor([[0., 1.], [1., 0.]]),
    ... )
    >>> fft2(x)
    (array([[ 6.,  0.],
        [ 0., -2.]], dtype=float32), array([[ 2.,  0.],
        [ 0., -2.]], dtype=float32))
    """
    if any_symbolic_tensors(x):
        return FFT2().symbolic_call(x)
    return backend.math.fft2(x)


class RFFT(Operation):
    def __init__(self, fft_length=None):
        super().__init__()
        self.fft_length = fft_length

    def compute_output_spec(self, x):
        # We are calculating 1D RFFT. Hence, rank >= 1.
        if len(x.shape) < 1:
            raise ValueError(
                f"Input should have rank >= 1. "
                f"Received: input.shape = {x.shape}"
            )

        if self.fft_length is not None:
            new_last_dimension = self.fft_length // 2 + 1
        else:
            if x.shape[-1] is not None:
                new_last_dimension = x.shape[-1] // 2 + 1
            else:
                new_last_dimension = None
        new_shape = x.shape[:-1] + (new_last_dimension,)

        return (
            KerasTensor(shape=new_shape, dtype=x.dtype),
            KerasTensor(shape=new_shape, dtype=x.dtype),
        )

    def call(self, x):
        return backend.math.rfft(x, fft_length=self.fft_length)


@keras_export("keras.ops.rfft")
def rfft(x, fft_length=None):
    """Real-valued Fast Fourier Transform along the last axis of the input.

    Computes the 1D Discrete Fourier Transform of a real-valued signal over the
    inner-most dimension of input.

    Since the Discrete Fourier Transform of a real-valued signal is
    Hermitian-symmetric, RFFT only returns the `fft_length / 2 + 1` unique
    components of the FFT: the zero-frequency term, followed by the
    `fft_length / 2` positive-frequency terms.

    Along the axis RFFT is computed on, if `fft_length` is smaller than the
    corresponding dimension of the input, the dimension is cropped. If it is
    larger, the dimension is padded with zeros.

    Args:
        x: Input tensor.
        fft_length: An integer representing the number of the fft length. If not
            specified, it is inferred from the length of the last axis of `x`.
            Defaults to `None`.

    Returns:
        A tuple containing two tensors - the real and imaginary parts of the
        output.

    Examples:

    >>> x = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> rfft(x)
    (array([10.0, -2.5, -2.5]), array([0.0, 3.4409548, 0.81229924]))

    >>> rfft(x, 3)
    (array([3.0, -1.5]), array([0.0, 0.8660254]))
    """
    if any_symbolic_tensors((x,)):
        return RFFT(fft_length).symbolic_call(x)
    return backend.math.rfft(x, fft_length)


class IRFFT(Operation):
    def __init__(self, fft_length=None):
        super().__init__()
        self.fft_length = fft_length

    def compute_output_spec(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                f"imaginary. Received: x={x}"
            )
        real, imag = x
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                "imaginary. Both the real and imaginary parts should have the "
                f"same shape. Received: x[0].shape = {real.shape}, "
                f"x[1].shape = {imag.shape}"
            )
        # We are calculating 1D IRFFT. Hence, rank >= 1.
        if len(real.shape) < 1:
            raise ValueError(
                f"Input should have rank >= 1. "
                f"Received: input.shape = {real.shape}"
            )

        if self.fft_length is not None:
            new_last_dimension = self.fft_length
        else:
            if real.shape[-1] is not None:
                new_last_dimension = 2 * (real.shape[-1] - 1)
            else:
                new_last_dimension = None
        new_shape = real.shape[:-1] + (new_last_dimension,)
        return KerasTensor(shape=new_shape, dtype=real.dtype)

    def call(self, x):
        return backend.math.irfft(x, fft_length=self.fft_length)


@keras_export("keras.ops.irfft")
def irfft(x, fft_length=None):
    """Inverse real-valued Fast Fourier transform along the last axis.

    Computes the inverse 1D Discrete Fourier Transform of a real-valued signal
    over the inner-most dimension of input.

    The inner-most dimension of the input is assumed to be the result of RFFT:
    the `fft_length / 2 + 1` unique components of the DFT of a real-valued
    signal. If `fft_length` is not provided, it is computed from the size of the
    inner-most dimension of the input `(fft_length = 2 * (inner - 1))`. If the
    FFT length used to compute is odd, it should be provided since it cannot
    be inferred properly.

    Along the axis IRFFT is computed on, if `fft_length / 2 + 1` is smaller than
    the corresponding dimension of the input, the dimension is cropped. If it is
    larger, the dimension is padded with zeros.

    Args:
        x: Tuple of the real and imaginary parts of the input tensor. Both
            tensors in the tuple should be of floating type.
        fft_length: An integer representing the number of the fft length. If not
            specified, it is inferred from the length of the last axis of `x`.
            Defaults to `None`.

    Returns:
        A tensor containing the inverse real-valued Fast Fourier Transform
        along the last axis of `x`.

    Examples:

    >>> real = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> imag = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> irfft((real, imag))
    array([0.66666667, -0.9106836, 0.24401694])

    >>> irfft(rfft(real, 5), 5)
    array([0.0, 1.0, 2.0, 3.0, 4.0])
    """
    if any_symbolic_tensors(x):
        return IRFFT(fft_length).symbolic_call(x)
    return backend.math.irfft(x, fft_length)


class STFT(Operation):
    def __init__(
        self,
        sequence_length,
        sequence_stride,
        fft_length,
        window="hann",
        center=True,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.fft_length = fft_length
        self.window = window
        self.center = center

    def compute_output_spec(self, x):
        if x.shape[-1] is not None:
            padded = 0 if self.center is False else (self.fft_length // 2) * 2
            num_sequences = (
                1
                + (x.shape[-1] + padded - self.fft_length)
                // self.sequence_stride
            )
        else:
            num_sequences = None
        new_shape = x.shape[:-1] + (num_sequences, self.fft_length // 2 + 1)
        return (
            KerasTensor(shape=new_shape, dtype=x.dtype),
            KerasTensor(shape=new_shape, dtype=x.dtype),
        )

    def call(self, x):
        return backend.math.stft(
            x,
            sequence_length=self.sequence_length,
            sequence_stride=self.sequence_stride,
            fft_length=self.fft_length,
            window=self.window,
            center=self.center,
        )


@keras_export("keras.ops.stft")
def stft(
    x, sequence_length, sequence_stride, fft_length, window="hann", center=True
):
    """Short-Time Fourier Transform along the last axis of the input.

    The STFT computes the Fourier transform of short overlapping windows of the
    input. This giving frequency components of the signal as they change over
    time.

    Args:
        x: Input tensor.
        sequence_length: An integer representing the sequence length.
        sequence_stride: An integer representing the sequence hop size.
        fft_length: An integer representing the size of the FFT to apply. If not
            specified, uses the smallest power of 2 enclosing `sequence_length`.
        window: A string, a tensor of the window or `None`. If `window` is a
            string, available values are `"hann"` and `"hamming"`. If `window`
            is a tensor, it will be used directly as the window and its length
            must be `sequence_length`. If `window` is `None`, no windowing is
            used. Defaults to `"hann"`.
        center: Whether to pad `x` on both sides so that the t-th sequence is
            centered at time `t * sequence_stride`. Otherwise, the t-th sequence
            begins at time `t * sequence_stride`. Defaults to `True`.

    Returns:
        A tuple containing two tensors - the real and imaginary parts of the
        STFT output.

    Example:

    >>> x = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> stft(x, 3, 2, 3)
    (array([[0.75, -0.375],
       [3.75, -1.875],
       [5.25, -2.625]]), array([[0.0, 0.64951905],
       [0.0, 0.64951905],
       [0.0, -0.64951905]]))
    """
    if any_symbolic_tensors((x,)):
        return STFT(
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
            fft_length=fft_length,
            window=window,
            center=center,
        ).symbolic_call(x)
    return backend.math.stft(
        x,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        fft_length=fft_length,
        window=window,
        center=center,
    )


class ISTFT(Operation):
    def __init__(
        self,
        sequence_length,
        sequence_stride,
        fft_length,
        length=None,
        window="hann",
        center=True,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.fft_length = fft_length
        self.length = length
        self.window = window
        self.center = center

    def compute_output_spec(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                f"imaginary. Received: x={x}"
            )
        real, imag = x
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                "imaginary. Both the real and imaginary parts should have the "
                f"same shape. Received: x[0].shape = {real.shape}, "
                f"x[1].shape = {imag.shape}"
            )
        if len(real.shape) < 2:
            raise ValueError(
                f"Input should have rank >= 2. "
                f"Received: input.shape = {real.shape}"
            )
        if real.shape[-2] is not None:
            output_size = (
                real.shape[-2] - 1
            ) * self.sequence_stride + self.fft_length
            if self.length is not None:
                output_size = self.length
            elif self.center:
                output_size = output_size - (self.fft_length // 2) * 2
        else:
            output_size = None
        new_shape = real.shape[:-2] + (output_size,)
        return KerasTensor(shape=new_shape, dtype=real.dtype)

    def call(self, x):
        return backend.math.istft(
            x,
            sequence_length=self.sequence_length,
            sequence_stride=self.sequence_stride,
            fft_length=self.fft_length,
            length=self.length,
            window=self.window,
            center=self.center,
        )


@keras_export("keras.ops.istft")
def istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
):
    """Inverse Short-Time Fourier Transform along the last axis of the input.

    To reconstruct an original waveform, the parameters should be the same in
    `stft`.

    Args:
        x: Tuple of the real and imaginary parts of the input tensor. Both
            tensors in the tuple should be of floating type.
        sequence_length: An integer representing the sequence length.
        sequence_stride: An integer representing the sequence hop size.
        fft_length: An integer representing the size of the FFT that produced
            `stft`.
        length: An integer representing the output is clipped to exactly length.
            If not specified, no padding or clipping take place. Defaults to
            `None`.
        window: A string, a tensor of the window or `None`. If `window` is a
            string, available values are `"hann"` and `"hamming"`. If `window`
            is a tensor, it will be used directly as the window and its length
            must be `sequence_length`. If `window` is `None`, no windowing is
            used. Defaults to `"hann"`.
        center: Whether `x` was padded on both sides so that the t-th sequence
            is centered at time `t * sequence_stride`. Defaults to `True`.

    Returns:
        A tensor containing the inverse Short-Time Fourier Transform along the
        last axis of `x`.

    Example:

    >>> x = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> istft(stft(x, 1, 1, 1), 1, 1, 1)
    array([0.0, 1.0, 2.0, 3.0, 4.0])
    """
    if any_symbolic_tensors(x):
        return ISTFT(
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
            fft_length=fft_length,
            window=window,
            center=center,
        ).symbolic_call(x)
    return backend.math.istft(
        x,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        fft_length=fft_length,
        length=length,
        window=window,
        center=center,
    )


class Rsqrt(Operation):
    def call(self, x):
        x = backend.convert_to_tensor(x)
        return backend.math.rsqrt(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)


@keras_export("keras.ops.rsqrt")
def rsqrt(x):
    """Computes reciprocal of square root of x element-wise.

    Args:
        x: input tensor

    Returns:
        A tensor with the same dtype as `x`.

    Example:

    >>> x = keras.ops.convert_to_tensor([1.0, 10.0, 100.0])
    >>> keras.ops.rsqrt(x)
    array([1.0, 0.31622776, 0.1], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Rsqrt().symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return backend.math.rsqrt(x)


class Erf(Operation):
    def compute_output_spec(self, x):
        return KerasTensor(shape=x.shape, dtype=x.dtype)

    def call(self, x):
        return backend.math.erf(x)


@keras_export("keras.ops.erf")
def erf(x):
    """Computes the error function of `x`, element-wise.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same dtype as `x`.

    Example:

    >>> x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
    >>> keras.ops.erf(x)
    array([-0.99998 , -0.99532, -0.842701,  0.,  0.842701], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Erf().symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return backend.math.erf(x)


class Erfinv(Operation):
    def compute_output_spec(self, x):
        return KerasTensor(shape=x.shape, dtype=x.dtype)

    def call(self, x):
        return backend.math.erfinv(x)


@keras_export("keras.ops.erfinv")
def erfinv(x):
    """Computes the inverse error function of `x`, element-wise.

    Args:
        x: Input tensor.

    Returns:
        A tensor with the same dtype as `x`.

    Example:

    >>> x = np.array([-0.5, -0.2, -0.1, 0.0, 0.3])
    >>> keras.ops.erfinv(x)
    array([-0.47694, -0.17914, -0.08886,  0. ,  0.27246], dtype=float32)
    """
    if any_symbolic_tensors((x,)):
        return Erfinv().symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return backend.math.erfinv(x)
