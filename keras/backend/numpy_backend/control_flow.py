import numpy as np

from keras.backend import floatx


def rnn(x, w, init, go_backwards=False, mask=None, unroll=False, input_length=None):
    w_i, w_h, w_o = w
    h = []
    o = []

    if go_backwards:
        t_list = range(x.shape[1] - 1, -1, -1)
    else:
        t_list = range(x.shape[1])

    if mask is not None:
        from keras import backend as K
        np_mask = K.eval(mask)
    else:
        np_mask = None

    for (i, t) in enumerate(t_list):
        h_t = np.dot(x[:, t], w_i)

        if w_h is not None:
            prev = h[i - 1] if i > 0 else init
            h_t1 = np.dot(prev, w_h)
            if np_mask is not None:
                h_t1[np_mask[:, t] == 0] = prev[np_mask[:, t] == 0]
        else:
            h_t1 = 0

        o_t = h_t + h_t1
        if w_o is not None:
            o_t = np.dot(o_t, w_o)
        o.append(o_t)

        if np_mask is not None:
            h_t = h_t * np_mask[:, t].reshape(-1, 1)
        h.append(h_t + h_t1)

    return o[-1], np.stack(o, axis=1), np.stack(h, axis=1)


def switch(condition, then_expression, else_expression):
    cond_float = condition.astype(floatx())
    while cond_float.ndim < then_expression.ndim:
        cond_float = cond_float[..., None]
    return cond_float * then_expression + (1 - cond_float) * else_expression


_LEARNING_PHASE = True


def learning_phase():
    return _LEARNING_PHASE


def set_learning_phase(value):
    global _LEARNING_PHASE
    _LEARNING_PHASE = value


def in_train_phase(x, alt, training=None):
    if training is None:
        training = learning_phase()

    if training is 1 or training is True:
        if callable(x):
            return x()
        else:
            return x
    else:
        if callable(alt):
            return alt()
        else:
            return alt


def in_test_phase(x, alt, training=None):
    return in_train_phase(alt, x, training=training)
