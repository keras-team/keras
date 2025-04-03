import types

from keras.src.distribution import distribution_lib
from keras.src.trainers.data_adapters import array_data_adapter
from keras.src.trainers.data_adapters import data_adapter
from keras.src.trainers.data_adapters import py_dataset_adapter
from keras.src.trainers.data_adapters.array_data_adapter import ArrayDataAdapter
from keras.src.trainers.data_adapters.generator_data_adapter import (
    GeneratorDataAdapter,
)
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDatasetAdapter
from keras.src.trainers.data_adapters.tf_dataset_adapter import TFDatasetAdapter
from keras.src.trainers.data_adapters.torch_data_loader_adapter import (
    TorchDataLoaderAdapter,
)


def get_data_adapter(
    x,
    y=None,
    sample_weight=None,
    batch_size=None,
    steps_per_epoch=None,
    shuffle=False,
    class_weight=None,
):
    # Allow passing a custom data adapter.
    if isinstance(x, data_adapter.DataAdapter):
        return x

    # Check for multi-process/worker distribution. Since only tf.dataset
    # is supported at the moment, we will raise error if the inputs fail
    # the type check
    distribution = distribution_lib.distribution()
    if getattr(distribution, "_is_multi_process", False) and not is_tf_dataset(
        x
    ):
        raise ValueError(
            "When using multi-worker distribution, the data must be provided "
            f"as a `tf.data.Dataset` instance. Received: type(x)={type(x)}."
        )

    if array_data_adapter.can_convert_arrays((x, y, sample_weight)):
        return ArrayDataAdapter(
            x,
            y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            shuffle=shuffle,
            batch_size=batch_size,
            steps=steps_per_epoch,
        )
    elif is_tf_dataset(x):
        # Unsupported args: y, sample_weight, shuffle
        if y is not None:
            raise_unsupported_arg("y", "the targets", "tf.data.Dataset")
        if sample_weight is not None:
            raise_unsupported_arg(
                "sample_weights", "the sample weights", "tf.data.Dataset"
            )
        return TFDatasetAdapter(
            x, class_weight=class_weight, distribution=distribution
        )
        # TODO: should we warn or not?
        # warnings.warn(
        #     "`shuffle=True` was passed, but will be ignored since the "
        #     "data `x` was provided as a tf.data.Dataset. The Dataset is "
        #     "expected to already be shuffled "
        #     "(via `.shuffle(tf.data.AUTOTUNE)`)"
        # )
    elif isinstance(x, py_dataset_adapter.PyDataset):
        if y is not None:
            raise_unsupported_arg("y", "the targets", "PyDataset")
        if sample_weight is not None:
            raise_unsupported_arg(
                "sample_weights", "the sample weights", "PyDataset"
            )
        return PyDatasetAdapter(x, class_weight=class_weight, shuffle=shuffle)
        # TODO: should we warn or not?
        # if x.num_batches is None and shuffle:
        #     warnings.warn(
        #         "`shuffle=True` was passed, but will be ignored since the "
        #         "data `x` was provided as a infinite PyDataset. The "
        #         "PyDataset is expected to already be shuffled."
        # )
    elif is_torch_dataloader(x):
        if y is not None:
            raise_unsupported_arg("y", "the targets", "torch DataLoader")
        if sample_weight is not None:
            raise_unsupported_arg(
                "sample_weights", "the sample weights", "torch DataLoader"
            )
        if class_weight is not None:
            raise ValueError(
                "Argument `class_weight` is not supported for torch "
                f"DataLoader inputs. Received: class_weight={class_weight}"
            )
        return TorchDataLoaderAdapter(x)
        # TODO: should we warn or not?
        # warnings.warn(
        #     "`shuffle=True` was passed, but will be ignored since the "
        #     "data `x` was provided as a torch DataLoader. The DataLoader "
        #     "is expected to already be shuffled."
        # )
    elif isinstance(x, types.GeneratorType):
        if y is not None:
            raise_unsupported_arg("y", "the targets", "PyDataset")
        if sample_weight is not None:
            raise_unsupported_arg(
                "sample_weights", "the sample weights", "PyDataset"
            )
        if class_weight is not None:
            raise ValueError(
                "Argument `class_weight` is not supported for Python "
                f"generator inputs. Received: class_weight={class_weight}"
            )
        return GeneratorDataAdapter(x)
        # TODO: should we warn or not?
        # warnings.warn(
        #     "`shuffle=True` was passed, but will be ignored since the "
        #     "data `x` was provided as a generator. The generator "
        #     "is expected to yield already-shuffled data."
        # )
    else:
        raise ValueError(f"Unrecognized data type: x={x} (of type {type(x)})")


def raise_unsupported_arg(arg_name, arg_description, input_type):
    raise ValueError(
        f"When providing `x` as a {input_type}, `{arg_name}` "
        f"should not be passed. Instead, {arg_description} should "
        f"be included as part of the {input_type}."
    )


def is_tf_dataset(x):
    if hasattr(x, "__class__"):
        for parent in x.__class__.__mro__:
            if parent.__name__ in (
                "DatasetV2",
                "DistributedDataset",
                "DistributedDatasetsFromFunction",
            ) and "tensorflow.python." in str(parent.__module__):
                return True
    return False


def is_torch_dataloader(x):
    if hasattr(x, "__class__"):
        for parent in x.__class__.__mro__:
            if parent.__name__ == "DataLoader" and "torch.utils.data" in str(
                parent.__module__
            ):
                return True
    return False
