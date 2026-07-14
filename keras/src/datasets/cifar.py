"""Utilities common to CIFAR10 and CIFAR100 datasets."""

from keras.src.datasets.npz_utils import RestrictedUnpickler


def load_batch(fpath, label_key="labels"):
    """Internal utility for parsing CIFAR data.

    Args:
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    Returns:
        A tuple `(data, labels)`.
    """
    with open(fpath, "rb") as f:
        # The CIFAR `data_batch_*`/`test_batch` files are pickle streams, so a
        # tampered file could execute code through a `__reduce__` gadget. Read
        # them with the restricted unpickler that only permits numpy array
        # reconstruction (the only globals a genuine batch needs).
        d = RestrictedUnpickler(f, encoding="bytes").load()
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode("utf8")] = v
        d = d_decoded
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels
