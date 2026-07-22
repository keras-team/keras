import os

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_TORCH_DEVICE"] = "cpu"
# Fallback for MPS if any op is missing
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def log(rank, msg):
    print(f"[RANK {rank}] {msg}", flush=True)


def run_dp_test(rank, world_size):
    try:
        import keras_hub

        import keras
        import keras.distribution

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        log(rank, "Initializing Data Parallel")
        keras.distribution.initialize()

        devices = [f"cpu:{i}" for i in range(world_size)]
        mesh = keras.distribution.DeviceMesh(
            shape=(world_size,), axis_names=("batch",), devices=devices
        )

        distribution = keras.distribution.DataParallel(device_mesh=mesh)

        with distribution.scope():
            log(rank, "Creating OPT model (DP)")
            model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")

            # Log model type to check for DDP wrapper
            # In Keras Torch backend, model.fit will wrap it in DDP
            model.compile(
                optimizer="adam", loss="sparse_categorical_crossentropy"
            )

            x = {
                "token_ids": np.random.randint(0, 50272, (4, 32)).astype(
                    "int32"
                ),
                "padding_mask": np.ones((4, 32), dtype="int32"),
            }
            y = np.random.randint(0, 768, (4, 32)).astype("int32")

            log(rank, "Starting fit (DP)")
            model.fit(x, y, epochs=1, verbose=0)

            # Check if ddp_model was used
            if hasattr(model, "ddp_model"):
                log(
                    rank,
                    f"SUCCESS: Model was wrapped in {type(model.ddp_model)}",
                )
            else:
                log(rank, "WARNING: model.ddp_model not found after fit")

        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        log(rank, f"DP TEST FAILED: {e}")
        import traceback

        traceback.print_exc()


def run_mp_test(rank, world_size):
    try:
        import keras_hub

        import keras
        import keras.distribution

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29506"

        log(rank, "Initializing Model Parallel")
        keras.distribution.initialize()

        devices = [f"cpu:{i}" for i in range(world_size)]
        mesh = keras.distribution.DeviceMesh(
            shape=(world_size,), axis_names=("model",), devices=devices
        )

        layout_map = keras.distribution.LayoutMap(mesh)
        # Shard embeddings and some kernels
        layout_map["token_embedding/embeddings"] = (
            keras.distribution.TensorLayout(("model", None), mesh)
        )

        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name=None
        )

        with distribution.scope():
            log(rank, "Creating OPT model (MP)")
            model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")

            # Verify sharding of the embedding layer
            for w in model.weights:
                if "token_embedding/embeddings" in w.name:
                    val = w.value
                    # In Torch, DTensor might be in .data if it's a Parameter
                    actual_tensor = (
                        val.data if isinstance(val, torch.nn.Parameter) else val
                    )
                    is_sharded = isinstance(actual_tensor, DTensor)
                    log(
                        rank,
                        f"Weight {w.name} is sharded (DTensor): {is_sharded}",
                    )
                    if is_sharded:
                        log(rank, f"  Placements: {actual_tensor.placements}")
                        log(
                            rank,
                            f"  Local shape: {actual_tensor.to_local().shape}",
                        )
                        log(rank, f"  Full shape: {actual_tensor.shape}")

            model.compile(
                optimizer="adam", loss="sparse_categorical_crossentropy"
            )

            x = {
                "token_ids": np.random.randint(0, 50272, (2, 32)).astype(
                    "int32"
                ),
                "padding_mask": np.ones((2, 32), dtype="int32"),
            }
            y = np.random.randint(0, 768, (2, 32)).astype("int32")

            log(rank, "Starting fit (MP)")
            model.fit(x, y, epochs=1, verbose=0)
            log(rank, "Fit completed (MP)")

        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        log(rank, f"MP TEST FAILED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    world_size = 2
    print("--- STARTING DATA PARALLEL TEST ---")
    torch.multiprocessing.spawn(
        run_dp_test, args=(world_size,), nprocs=world_size, join=True
    )

    print("--- STARTING MODEL PARALLEL TEST ---")
    torch.multiprocessing.spawn(
        run_mp_test, args=(world_size,), nprocs=world_size, join=True
    )
