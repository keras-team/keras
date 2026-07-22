import os
import sys

import torch
import torch.distributed as dist

# Import and configure keras backend FIRST before any other imports
os.environ["KERAS_BACKEND"] = "torch"


def _test_fn(rank, world_size):
    try:
        # CRITICAL: Add local Keras code to path in EACH spawned process
        # so sys.path modifications in parent won't be inherited
        import numpy as np

        sys.path.insert(0, "/Users/suhanaaa/keras")

        # Now import keras (local fixed version) and keras-hub (preinstalled)
        import keras_hub

        import keras
        import keras.distribution

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

        # Minimal sharding layout for Gemma (Embeddings only)
        layout_map = keras.distribution.LayoutMap(mesh)

        # Shard embeddings (to test arange/position promotion and cast)
        layout_map["token_embedding/embeddings"] = (
            keras.distribution.TensorLayout(("model", None), mesh)
        )

        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map,
            batch_dim_name="model",
            auto_shard_dataset=False,
        )

        with distribution.scope():
            print(f"[PROCESS {rank}] Creating model")
            # Create a small Gemma 2 style backbone to save memory
            model = keras_hub.models.GemmaBackbone(
                vocabulary_size=1000,
                num_layers=2,
                num_query_heads=4,
                num_key_value_heads=1,
                hidden_dim=256,
                intermediate_dim=512,
                head_dim=64,
                attention_logit_soft_cap=50,
                final_logit_soft_cap=30,
                use_sliding_window_attention=True,
            )

            # Compile model
            model.compile(
                optimizer="adam", loss="sparse_categorical_crossentropy"
            )

            # Create dummy data
            np.random.seed(42 + rank)
            x = {
                "token_ids": np.random.randint(0, 1000, (2, 32)).astype(
                    "int32"
                ),
                "padding_mask": np.ones((2, 32), dtype="int32"),
            }
            y = np.random.randint(0, 256, (2, 32)).astype("int32")

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
