import numpy as np

from keras.src.distribution.tensor_parallel.communications import (
    TensorParallelCommunicator,
)

communicator = TensorParallelCommunicator(world_size=4, rank=0)


def test_slice_gradient_for_column_parallel_even_division():
    """Tests slicing when the dimension is evenly divisible by world_size."""
    world_size = 4
    full_gradient = np.arange(16).reshape(1, 16)

    sliced_gradient = communicator.slice_upstream_gradient_for_column_parallel(
        full_gradient, rank=2, world_size=world_size, dim=-1
    )

    expected_slice = np.array([[8, 9, 10, 11]])
    np.testing.assert_array_equal(sliced_gradient, expected_slice)
    assert sliced_gradient.shape == (1, 4)


def test_slice_gradient_for_column_parallel_uneven_division():
    """Tests slicing with a remainder, which gets distributed to early ranks."""
    world_size = 4
    full_gradient = np.arange(17).reshape(1, 17)

    slice_rank_0 = communicator.slice_upstream_gradient_for_column_parallel(
        full_gradient, rank=0, world_size=world_size, dim=-1
    )
    assert slice_rank_0.shape == (1, 5)
    np.testing.assert_array_equal(slice_rank_0, np.array([[0, 1, 2, 3, 4]]))

    slice_rank_1 = communicator.slice_upstream_gradient_for_column_parallel(
        full_gradient, rank=1, world_size=world_size, dim=-1
    )
    assert slice_rank_1.shape == (1, 4)
    np.testing.assert_array_equal(slice_rank_1, np.array([[5, 6, 7, 8]]))


def test_slice_gradient_for_row_parallel():
    """Tests the simpler slicing logic for row-parallel."""
    world_size = 4
    full_gradient = np.arange(16).reshape(16, 1)
    sliced_gradient = communicator.slice_upstream_gradient_for_row_parallel(
        full_gradient, rank=3, world_size=world_size, dim=0
    )

    expected_slice = np.array([[12], [13], [14], [15]])
    np.testing.assert_array_equal(sliced_gradient, expected_slice)
    assert sliced_gradient.shape == (4, 1)
