import numpy as np


# def get_test_data(
#     train_samples, test_samples, input_shape, num_classes, random_seed=None
# ):
#     """Generates test data to train a model on.

#     Args:
#         train_samples: Integer, how many training samples to generate.
#         test_samples: Integer, how many test samples to generate.
#         input_shape: Tuple of integers, shape of the inputs.
#         num_classes: Integer, number of classes for the data and targets.
#         random_seed: Integer, random seed used by Numpy to generate data.

#     Returns:
#         A tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
#     """
#     if random_seed is not None:
#         np.random.seed(random_seed)
#     num_sample = train_samples + test_samples
#     templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
#     y = np.random.randint(0, num_classes, size=(num_sample,))
#     x = np.zeros((num_sample,) + input_shape, dtype=np.float32)
#     for i in range(num_sample):
#         x[i] = templates[y[i]] + np.random.normal(
#             loc=0, scale=1.0, size=input_shape
#         )
#     x_train, y_train, x_test, y_test = (
#         x[:train_samples],
#         y[:train_samples],
#         x[train_samples:],
#         y[train_samples:],
#     )

#     return (x_train, y_train, x_test, y_test)


def get_test_data(
    train_samples, test_samples, input_shape, num_classes, random_seed=None
):
    """
    Generates synthetic test data for training a model, ensuring balanced class distribution in train/test split.

    Args:
        train_samples (int): Number of training samples.
        test_samples (int): Number of testing samples.
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for the data and targets.
        random_seed (int, optional): Random seed for data generation. Defaults to None.

    Returns:
        tuple: Four numpy arrays representing training data, training labels, test data, and test labels.
    """
    np.random.seed(random_seed)

    # Total samples
    total_samples = train_samples + test_samples

    # Ensure that we generate a balanced dataset
    samples_per_class = total_samples // num_classes
    y = np.array(
        [i for i in range(num_classes) for _ in range(samples_per_class)]
    )

    # Generate extra samples if needed due to rounding
    extra_samples = total_samples - len(y)
    y_extra = np.random.randint(0, num_classes, size=(extra_samples,))
    y = np.concatenate([y, y_extra])

    # Generate data
    templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
    x = np.zeros((total_samples,) + input_shape, dtype=np.float32)
    for i in range(total_samples):
        x[i] = templates[y[i]] + np.random.normal(
            loc=0, scale=1.0, size=input_shape
        )

    # Shuffle data and split
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    x, y = x[indices], y[indices]
    x_train, x_test = x[:train_samples], x[train_samples:]
    y_train, y_test = y[:train_samples], y[train_samples:]

    return x_train, y_train, x_test, y_test
