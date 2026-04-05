import os

import keras_hub
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
        os.environ["MASTER_PORT"] = "29507"

        print(f"[PROCESS {rank}] Initializing distribution")
        keras.distribution.initialize()

        # Use CPU for this simulation
        devices = [f"cpu:{i}" for i in range(world_size)]

        mesh = keras.distribution.DeviceMesh(
            shape=(world_size,), axis_names=("batch",), devices=devices
        )

        # DataParallel strategy
        distribution = keras.distribution.DataParallel(
            device_mesh=mesh, auto_shard_dataset=False
        )

        with distribution.scope():
            print(f"[PROCESS {rank}] Creating model from preset")
            model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")

            # Compile model
            model.compile(
                optimizer="adam", loss="sparse_categorical_crossentropy"
            )

            # Create dummy data
            np.random.seed(42 + rank)
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


def test_data_parallel_fit():
    world_size = 2
    print(f"Starting DP test with world_size={world_size}")
    torch.multiprocessing.spawn(
        _test_fn, args=(world_size,), nprocs=world_size, join=True
    )


if __name__ == "__main__":
    test_data_parallel_fit()
