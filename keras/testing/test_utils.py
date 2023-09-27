import numpy as np


def get_test_data(
    train_samples, test_samples, input_shape, num_classes, random_seed=None
):
    """Generates synthetic test data for training a model,
    ensuring balanced class distribution in train/test split
    using stratified sampling.

    Args:
        train_samples (int): Number of training samples.
        test_samples (int): Number of testing samples.
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for the data and targets.
        random_seed (int, optional): Random seed for data generation.
                                    Defaults to None.

    Returns:
        tuple: Four numpy arrays representing training data,
               training labels, test data, and test labels.
    """
    np.random.seed(random_seed)

    # Total samples
    total_samples = train_samples + test_samples

    # Ensure that we generate a balanced dataset
    samples_per_class = total_samples // num_classes
    y = np.array(
        [i for i in range(num_classes) for _ in range(samples_per_class)],
        dtype=np.int64,  # Ensure integer datatype as np.int64
    )

    # Generate extra samples in a deterministic manner
    extra_samples = total_samples - len(y)
    y_extra = np.array(
        [i % num_classes for i in range(extra_samples)], dtype=np.int64
    )
    y = np.concatenate([y, y_extra])

    # Generate data
    templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
    x = np.zeros((total_samples,) + input_shape, dtype=np.float32)
    for i in range(total_samples):
        x[i] = templates[y[i]] + np.random.normal(
            loc=0, scale=1.0, size=input_shape
        )

    # Shuffle the entire dataset to ensure randomness based on seed
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    x, y = x[indices], y[indices]

    # Stratified Shuffle Split
    x_train, y_train, x_test, y_test = [], [], [], []
    for cls in range(num_classes):
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        train_count = int(train_samples / num_classes)

        x_train.extend(x[cls_indices[:train_count]])
        y_train.extend(y[cls_indices[:train_count]])

        x_test.extend(x[cls_indices[train_count:]])
        y_test.extend(y[cls_indices[train_count:]])

    # Convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Shuffle training and test sets after stratified split
    train_indices = np.arange(len(x_train))
    test_indices = np.arange(len(x_test))
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    x_train, y_train = x_train[train_indices], y_train[train_indices]
    x_test, y_test = x_test[test_indices], y_test[test_indices]

    return x_train, y_train, x_test, y_test
