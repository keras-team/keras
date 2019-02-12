import numpy as np

from keras.initializers import Orthogonal


def test_orthogonal_init_does_not_affect_global_rng():
    np.random.seed(1337)
    before = np.random.randint(0, 100, size=10)

    np.random.seed(1337)
    init = Orthogonal(seed=9876)
    init(shape=(10, 5))
    after = np.random.randint(0, 100, size=10)

    assert np.array_equal(before, after)
