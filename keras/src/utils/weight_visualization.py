import matplotlib.pyplot as plt
import numpy as np


def find_factors_closest_to_sqrt(num):
    sqrt_num = int(np.sqrt(num))

    for i in range(sqrt_num, 0, -1):
        if num % i == 0:
            M = i
            N = num // i
            return M, N


def reduce_dim(value):
    num_filters = value.shape[-1]

    width, height = find_factors_closest_to_sqrt(num_filters)

    width_values = []
    for i in range(height):
        width_val = value[..., i * width : (i + 1) * width]
        width_val = np.hstack(width_val)
        width_values.append(width_val)

    width_values = np.asarray(width_values)
    ret = np.vstack(width_values)
    return ret


def reduce_dim_to_2d(w_value):
    if len(w_value.shape) > 2:
        w_value = reduce_dim(w_value)

    if len(w_value.shape) == 2:
        return w_value

    return reduce_dim_to_2d(w_value)


def downscale_arr(arr, downscale_threshold):
    W = arr.shape[0]
    H = arr.shape[1]

    if W > downscale_threshold:
        if W % 2 == 0:
            arr = arr.reshape(W // 2, 2, H, 1).mean(axis=(1, 3))
        elif W % 3 == 0:
            arr = arr.reshape(W // 3, 3, H, 1).mean(axis=(1, 3))
        W = arr.shape[0]

    if H > downscale_threshold:
        if H % 2 == 0:
            arr = arr.reshape(W, 1, H // 2, 2).mean(axis=(1, 3))
        elif H % 3 == 0:
            arr = arr.reshape(W, 1, H // 3, 3).mean(axis=(1, 3))

    return arr


def make_dim_to_2d(w_value):
    num_filters = w_value.shape[0]
    width, height = find_factors_closest_to_sqrt(num_filters)
    ret = w_value.numpy().reshape(width, height)

    return ret


def plot_weight(weight, downscale_threshold=None):
    w_value = weight.value

    if len(w_value.shape) > 2:
        w_value = reduce_dim_to_2d(w_value)
    else:
        w_value = make_dim_to_2d(w_value)

    w_min, w_max = np.min(w_value), np.max(w_value)
    w_value = (w_value - w_min) / (w_max - w_min)

    if downscale_threshold is not None:
        while True:
            if max(w_value.shape) < downscale_threshold:
                break

            w_value = downscale_arr(w_value, downscale_threshold)

    plt.figure(figsize=(7, 7))
    plt.suptitle(weight.path)
    plt.imshow(w_value, cmap="bwr")
    plt.show()
