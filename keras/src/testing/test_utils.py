import numpy as np


def get_test_data(
    train_samples, test_samples, input_shape, num_classes, random_seed=None
):
    """Generates balanced, stratified synthetic test data to train a model on.

    Args:
        train_samples: Integer, how many training samples to generate.
        test_samples: Integer, how many test samples to generate.
        input_shape: Tuple of integers, shape of the inputs.
        num_classes: Integer, number of classes for the data and targets.
        random_seed: Integer, random seed used by Numpy to generate data.

    Returns:
        A tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    np.random.seed(random_seed)

    # Total samples
    total_samples = train_samples + test_samples

    # Ensure that we generate a balanced dataset
    samples_per_class = total_samples // num_classes
    y = np.array(
        [i for i in range(num_classes) for _ in range(samples_per_class)],
        dtype=np.int32,
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

    return (x_train, y_train), (x_test, y_test)


def named_product(*args, **kwargs):
    """Utility to generate the cartesian product of parameters values and
    generate a test case names for each combination.

    The result of this function is to be used with the
    `@parameterized.named_parameters` decorator. It is a replacement for
    `@parameterized.product` which adds explicit test case names.

    For example, this code:
    ```
    class NamedExample(parameterized.TestCase):
        @parameterized.named_parameters(
            named_product(
                [
                    {'testcase_name': 'negative', 'x': -1},
                    {'testcase_name': 'positive', 'x': 1},
                    {'testcase_name': 'zero', 'x': 0},
                ],
                numeral_type=[float, int],
            )
        )
        def test_conversion(self, x, numeral_type):
            self.assertEqual(numeral_type(x), x)
    ```
    produces six tests (note that absl will reorder them by name):
    - `NamedExample::test_conversion_negative_float`
    - `NamedExample::test_conversion_positive_float`
    - `NamedExample::test_conversion_zero_float`
    - `NamedExample::test_conversion_negative_int`
    - `NamedExample::test_conversion_positive_int`
    - `NamedExample::test_conversion_zero_int`

    This function is also useful in the case where there is no product to
    generate test case names for one argument:
    ```
    @parameterized.named_parameters(named_product(numeral_type=[float, int]))
    ```

    Args:
        *args: Each positional parameter is a sequence of keyword arg dicts.
            Every test case generated will include exactly one dict from each
            positional parameter. These will then be merged to form an overall
            list of arguments for the test case. Each dict must contain a
            `"testcase_name"` key whose value is combined with others to
            generate the test case name.
        **kwargs: A mapping of parameter names and their possible values.
            Possible values should given as either a list or a tuple. A string
            representation of each value is used to generate the test case name.

    Returns:
        A list of maps for the test parameters combinations to pass to
        `@parameterized.named_parameters`.
    """

    def value_to_str(value):
        if hasattr(value, "__name__"):
            return value.__name__.lower()
        return str(value).lower()

    # Convert the keyword arguments in the same dict format as the args
    all_test_dicts = args + tuple(
        tuple({"testcase_name": value_to_str(v), key: v} for v in values)
        for key, values in kwargs.items()
    )

    # The current list of tests, start with one empty test
    tests = [{}]
    for test_dicts in all_test_dicts:
        new_tests = []
        for test_dict in test_dicts:
            for test in tests:
                # Augment the testcase name by appending
                testcase_name = test.get("testcase_name", "")
                testcase_name += "_" if testcase_name else ""
                testcase_name += test_dict["testcase_name"]
                new_test = test.copy()
                # Augment the test by adding all the parameters
                new_test.update(test_dict)
                new_test["testcase_name"] = testcase_name
                new_tests.append(new_test)
        # Overwrite the list of tests with the product obtained so far
        tests = new_tests

    return tests


def assert_symbolic_shape_consistency(
    test_case, layer, input_shape, expected_output_shape=None
):
    """Test utility to verify symbolic tensor shape computation consistency.

    This utility tests that a layer correctly computes output shapes for
    symbolic tensors and that the shapes are consistent with eager tensor
    execution.

    Args:
        test_case: TestCase instance to use for assertions.
        layer: Keras layer to test.
        input_shape: Tuple representing the input shape (without batch dim).
        expected_output_shape: Optional tuple representing expected output
            shape (without batch dimension). If None, only consistency is
            tested.
    """
    import numpy as np

    import keras

    # Create symbolic input
    symbolic_input = keras.Input(shape=input_shape)
    symbolic_output = layer(symbolic_input)
    # Create eager input with batch dimension
    batch_size = 2
    eager_input = np.random.uniform(-1, 1, size=(batch_size,) + input_shape)

    try:
        eager_output = layer(eager_input)
    except Exception as e:
        # If layer requires adaptation or other setup, skip eager test
        if "adapt" in str(e).lower() or "boundaries" in str(e).lower():
            # Only test symbolic shape
            if expected_output_shape is not None:
                expected_symbolic_shape = (None,) + expected_output_shape
                test_case.assertEqual(
                    symbolic_output.shape,
                    expected_symbolic_shape,
                    f"Symbolic shape mismatch: expected "
                    f"{expected_symbolic_shape}, got {symbolic_output.shape}",
                )
            return
        else:
            raise e
    # Test shape consistency between symbolic and eager
    symbolic_shape_no_batch = symbolic_output.shape[1:]
    eager_shape_no_batch = eager_output.shape[1:]
    test_case.assertEqual(
        symbolic_shape_no_batch,
        eager_shape_no_batch,
        f"Shape inconsistency: symbolic {symbolic_shape_no_batch} != "
        f"eager {eager_shape_no_batch}",
    )
    # Test expected output shape if provided
    if expected_output_shape is not None:
        test_case.assertEqual(
            symbolic_shape_no_batch,
            expected_output_shape,
            f"Output shape mismatch: expected {expected_output_shape}, "
            f"got {symbolic_shape_no_batch}",
        )
