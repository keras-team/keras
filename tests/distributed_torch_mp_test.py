import os

# Import and configure keras backend FIRST before any other imports
os.environ["KERAS_BACKEND"] = "torch"

# Force import of torch backend and distribution lib early to apply patches
import keras_hub
import numpy as np
import torch
import torch.distributed as dist


def _test_fn(rank, world_size):
    try:
        # Ensure patches are applied in this process
        import keras.src.backend.torch.distribution_lib  # noqa: trigger patches

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        print(f"[PROCESS {rank}] Initializing distribution")
        keras.distribution.initialize()

        # Define mesh and layout
        if torch.cuda.is_available():
            devices = keras.distribution.list_devices("gpu")
        else:
            devices = keras.distribution.list_devices("cpu")
            # Mock devices if not enough
            if len(devices) < world_size:
                devices = ["cpu:0"] * world_size

        mesh = keras.distribution.DeviceMesh(
            shape=(world_size,),
            axis_names=("model",),
            devices=devices[:world_size],
        )

        # ModelParallel strategy with non-overlapping patterns
        layout_map = keras.distribution.LayoutMap(mesh)

        # Keep embeddings replicated to avoid unbind issues
        # Use specific non-overlapping paths
        layout_map["embeddings/token_embedding/.*"] = (
            keras.distribution.TensorLayout((None, None), mesh)
        )
        layout_map["embeddings/position_embedding/.*"] = (
            keras.distribution.TensorLayout((None, None), mesh)
        )

        # Shard attention query/key/value projections for model parallelism
        layout_map[".*attention.*query.*kernel"] = (
            keras.distribution.TensorLayout(("model", None), mesh)
        )
        layout_map[".*attention.*key.*kernel"] = (
            keras.distribution.TensorLayout(("model", None), mesh)
        )
        layout_map[".*attention.*value.*kernel"] = (
            keras.distribution.TensorLayout(("model", None), mesh)
        )

        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map,
            batch_dim_name="model",
            auto_shard_dataset=False,
        )

        with distribution.scope():
            print(f"[PROCESS {rank}] Creating model")
            # Load OPT 125m model from preset as requested
            model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")

            # Compile model
            model.compile(
                optimizer="adam", loss="sparse_categorical_crossentropy"
            )

            # Create dummy data
            np.random.seed(42 + rank)
            # opt_125m has vocabulary_size=50272, max_sequence_length=2048
            x = {
                "token_ids": np.random.randint(0, 50272, (2, 32)).astype(
                    "int32"
                ),
                "padding_mask": np.ones((2, 32), dtype="int32"),
            }
            y = np.random.randint(0, 768, (2, 32)).astype("int32")

            print(f"[PROCESS {rank}] Starting fit")
            model.fit(x, y, epochs=1)
            print(f"[PROCESS {rank}] Fit completed")

        if torch.distributed.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"[PROCESS {rank}] FAILED with error: {e}")
        import traceback

        traceback.print_exc()
        try:
            if torch.distributed.is_initialized():
                dist.destroy_process_group()
        except:
            pass
        raise e


def test_model_parallel_fit():
    world_size = 2
    print(f"Starting test with world_size={world_size}")
    torch.multiprocessing.spawn(
        _test_fn, args=(world_size,), nprocs=world_size, join=True
    )


if __name__ == "__main__":
    test_model_parallel_fit()
