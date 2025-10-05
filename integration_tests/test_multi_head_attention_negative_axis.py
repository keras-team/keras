import numpy as np

import keras


def test_attention_axes_negative_indexing_matches_positive():
    x = np.random.normal(size=(2, 3, 8, 4))

    mha_pos = keras.layers.MultiHeadAttention(
        num_heads=2, key_dim=4, attention_axes=2
    )
    mha_neg = keras.layers.MultiHeadAttention(
        num_heads=2, key_dim=4, attention_axes=-2
    )

    _ = mha_pos(x, x)
    _ = mha_neg(x, x)

    mha_neg.set_weights(mha_pos.get_weights())

    z_pos, a_pos = mha_pos(x, x, return_attention_scores=True)
    z_neg, a_neg = mha_neg(x, x, return_attention_scores=True)

    assert z_pos.shape == z_neg.shape
    assert a_pos.shape == a_neg.shape

    np.testing.assert_allclose(z_pos, z_neg, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(a_pos, a_neg, rtol=1e-5, atol=1e-5)
