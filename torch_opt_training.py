import os
import torch

# Keras settings
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

# Set NCCL environment variables
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import keras
import keras_hub
from keras.distribution import DeviceMesh, LayoutMap, ModelParallel, TensorLayout
import torch.distributed as dist
import atexit

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

atexit.register(cleanup)

def train_opt_model_parallel():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    keras.distribution.initialize()

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = dist.get_world_size()

    # Device setup
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        devices = [f"cuda:{i}" for i in range(world_size)]
    else:
        device = "cpu"
        devices = [f"cpu:{i}" for i in range(world_size)]
    
    os.environ["KERAS_TORCH_DEVICE"] = device
    
    mesh_shape = (world_size,)
    mesh = DeviceMesh(shape=mesh_shape, axis_names=("model",), devices=devices)

    # ... (Rest of the layout map logic)

    tmp_model = keras_hub.models.OPTBackbone(
        vocabulary_size=1000, num_layers=2, num_heads=2, hidden_dim=64, intermediate_dim=128, max_sequence_length=32, dropout=0.0,
    )
    tmp_model.build({"token_ids": (None, 32), "padding_mask": (None, 32)})

    layout_map = LayoutMap(mesh)
    for var in tmp_model.variables:
        path = var.path
        axes = [None] * len(var.shape)
        
        # 1. Embeddings: Shard embedding dim
        if "embeddings" in path:
            axes[-1] = "model"
            
        # 2. FFN: Standard Colwise/Rowwise
        elif "feedforward_intermediate_dense/kernel" in path:
            axes[-1] = "model"
        elif "feedforward_intermediate_dense/bias" in path:
            axes[0] = "model"
        elif "feedforward_output_dense/kernel" in path:
            axes[0] = "model"
        elif "feedforward_output_dense/bias" in path:
            axes[0] = "model"
            
        # 3. LayerNorm: Shard hidden dim
        elif "layer_norm" in path and ("gamma" in path or "beta" in path):
            axes[0] = "model"
            
        # 4. Attention: Shard heads or hidden dim
        elif "self_attention" in path:
            if "/kernel" in path:
                if len(var.shape) == 3: # Multi-head kernels
                    axes[1] = "model"
                else: # attention_output kernel
                    axes[0] = "model"
            elif "/bias" in path:
                if len(var.shape) == 2: # Multi-head biases
                    axes[0] = "model"
                else: # attention_output bias
                    axes[0] = "model"
        
        # 5. Catch-all for any missed 1D or 2D variables
        if "model" not in axes:
            if len(var.shape) > 0:
                axes[0] = "model"
        
        layout = TensorLayout(axes=tuple(axes), device_mesh=mesh)
        layout_map[path] = layout

    print(f"Constructed LayoutMap with {len(layout_map._layout_map)} entries")
    print(f"  Testing end-to-end with permanent backend fixes (ALL LAYERS SHARDED)")

    distribution = ModelParallel(layout_map=layout_map, auto_shard_dataset=False)

    # Build model INSIDE distribution scope
    print(f"Creating and building model within distribution scope on RANK {rank}...")
    with distribution.scope():
        model = keras_hub.models.OPTBackbone(
            vocabulary_size=1000, num_layers=2, num_heads=2, hidden_dim=64, intermediate_dim=128, max_sequence_length=32, dropout=0.0,
        )
        model.build({"token_ids": (None, 32), "padding_mask": (None, 32)})
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=keras.losses.MeanSquaredError())
        
        # Prepare data and fit
        token_ids = np.random.randint(0, 1000, (16, 32)).astype("int32")
        padding_mask = np.ones((16, 32), dtype="int32")
        y = np.random.randn(16, 32, 64).astype("float32")
        x = {"token_ids": token_ids, "padding_mask": padding_mask}
        
        # Explicitly build the model before calling fit()
        dummy_input = {"token_ids": np.random.randint(0, 1000, (1, 32)).astype("int32"),
                       "padding_mask": np.ones((1, 32), dtype="int32")}
        model(dummy_input)

        history = model.fit(x, y, epochs=1, batch_size=4, verbose=1)
        print(f"\n✓ model.fit() completed successfully on RANK {rank}!")
        final_loss = float(history.history['loss'][-1])
        print(f"  Final loss on RANK {rank}: {final_loss:.4f}")
            
        # Validation
        print(f"Validating sharding on RANK {rank}...")
        shard_count = 0
        replicate_count = 0
        import torch.distributed.tensor as dt
        for v in model.variables:
            if hasattr(v.value, 'placements'):
                has_shard = any(isinstance(p, dt.Shard) for p in v.value.placements)
                if has_shard:
                    shard_count += 1
                    if rank == 0 and ("/kernel" in v.path or "layer_norm" in v.path or "token_embedding" in v.path):
                        print(f"  [SHARDED] {v.path}: {v.value.placements}")
                else: replicate_count += 1
            else: replicate_count += 1
        
        print(f"\nSharding Summary on RANK {rank}:")
        print(f"  Sharded variables: {shard_count}")
        print(f"  Replicated variables: {replicate_count}")

if __name__ == "__main__":
    train_opt_model_parallel()