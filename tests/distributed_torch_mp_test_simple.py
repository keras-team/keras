import os

import numpy as np
import torch
import torch.distributed as dist

import keras

# Set backend to torch
os.environ["KERAS_BACKEND"] = "torch"


def _test_fn(rank, world_size):
    try:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29506"

        print(f"[PROCESS {rank}] Initializing distribution")
        keras.distribution.initialize()

        # Use CPU for this simulation to avoid MPS/CUDA complexities
        devices = [f"cpu:{i}" for i in range(world_size)]

        mesh = keras.distribution.DeviceMesh(
            shape=(world_size,), axis_names=("model",), devices=devices
        )

        # ModelParallel strategy
        layout_map = keras.distribution.LayoutMap(mesh)
        # Shard the dense layer kernel on the "model" axis
        layout_map[".*dense.*kernel"] = keras.distribution.TensorLayout(
            ("model", None), mesh
        )

        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map,
            batch_dim_name="model",
            auto_shard_dataset=False,
        )

        with distribution.scope():
            print(f"[PROCESS {rank}] Creating model")
            model = keras.Sequential(
                [
                    keras.layers.Input(shape=(64,)),
                    keras.layers.Dense(128, name="dense_1"),
                    keras.layers.Dense(10, name="dense_2"),
                ]
            )

            # Compile model
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Create dummy data
            # Each process gets a local batch of 4 samples
            # Since auto_shard_dataset=False, we simulate each process having
            # its own data
            x = np.random.randn(4, 64).astype("float32")
            y = np.random.randint(0, 10, (4,)).astype("int32")

            print(f"[PROCESS {rank}] Starting fit")
            model.fit(x, y, epochs=1)
            print(f"[PROCESS {rank}] Fit completed")

            print(f"[PROCESS {rank}] Starting evaluate")
            model.evaluate(x, y)
            print(f"[PROCESS {rank}] Evaluate completed")

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
