import json
import os
import subprocess
import sys
import time

import numpy as np

# Suppress framework noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_backend(backend, world_size=2):
    os.environ["KERAS_BACKEND"] = backend
    if backend == "jax":
        os.environ["XLA_FLAGS"] = (
            f"--xla_force_host_platform_device_count={world_size} "
            "--xla_cpu_multi_thread_eigen=false "
            "intra_op_parallelism_threads=1"
        )
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        num_gpus = 0
        try:
            import torch

            num_gpus = torch.cuda.device_count()
        except:
            pass

        if num_gpus < world_size:
            os.environ["JAX_PLATFORMS"] = "cpu"
        _run_jax(world_size)

    elif backend == "torch":
        import torch

        port = str(find_free_port())
        torch.multiprocessing.spawn(
            _run_torch, args=(world_size, port), nprocs=world_size, join=True
        )


def _run_jax(world_size):
    import keras_hub

    import keras

    keras.utils.set_random_seed(42)

    devices = keras.distribution.list_devices()
    if len(devices) > world_size:
        devices = devices[:world_size]
    print(f"Using JAX devices: {devices}")

    mesh = keras.distribution.DeviceMesh(
        shape=(world_size,), axis_names=("batch",), devices=devices
    )
    distribution = keras.distribution.DataParallel(
        device_mesh=mesh, auto_shard_dataset=False
    )

    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset(
            "opt_125m_en", dropout=0.0
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="mse"
        )

        np.random.seed(42)
        base_batch_size = 16
        global_batch_size = base_batch_size * world_size

        num_samples = global_batch_size * 6
        x_full = {
            "token_ids": np.random.randint(0, 50272, (num_samples, 32)).astype(
                "int32"
            ),
            "padding_mask": np.ones((num_samples, 32), dtype="int32"),
        }
        y_full = np.random.normal(size=(num_samples, 32, 768)).astype("float32")

        # Warmup Step
        model.fit(
            x_full,
            y_full,
            batch_size=global_batch_size,
            epochs=1,
            steps_per_epoch=1,
            verbose=1,
            shuffle=False,
        )

        x_train = {k: v[global_batch_size:] for k, v in x_full.items()}
        y_train = y_full[global_batch_size:]

        start_time = time.time()
        epochs = 1
        steps_per_epoch = 5
        history = model.fit(
            x_train,
            y_train,
            batch_size=global_batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=1,
            shuffle=False,
        )
        end_time = time.time()

        final_loss = float(history.history["loss"][0])
        training_time = end_time - start_time
        total_samples = global_batch_size * steps_per_epoch * epochs
        throughput = total_samples / training_time
        perplexity = float(np.exp(final_loss))

    results = {
        "final_loss": final_loss,
        "perplexity": perplexity,
        "throughput": throughput,
    }
    with open("results_jax_dp.json", "w") as f:
        json.dump(results, f, indent=2)


def _run_torch(rank, world_size, port):
    import os

    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port

    num_gpus = torch.cuda.device_count()
    if num_gpus >= world_size:
        os.environ["KERAS_TORCH_DEVICE"] = f"cuda:{rank}"
        device_type = "cuda"
    else:
        os.environ["KERAS_TORCH_DEVICE"] = "cpu"
        device_type = "cpu"

    import sys

    import keras_hub

    import keras

    sys.path.insert(0, os.getcwd())
    keras.utils.set_random_seed(42)
    keras.distribution.initialize()

    devices = keras.distribution.list_devices(device_type)[:world_size]
    print(f"[Rank {rank}] Using devices: {devices}")

    mesh = keras.distribution.DeviceMesh(
        shape=(world_size,), axis_names=("batch",), devices=devices
    )
    distribution = keras.distribution.DataParallel(
        device_mesh=mesh, auto_shard_dataset=False
    )

    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset(
            "opt_125m_en", dropout=0.0
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="mse"
        )

        np.random.seed(42)
        base_batch_size = 16
        global_batch_size = base_batch_size * world_size

        num_samples = global_batch_size * 6
        x_full = {
            "token_ids": np.random.randint(0, 50272, (num_samples, 32)).astype(
                "int32"
            ),
            "padding_mask": np.ones((num_samples, 32), dtype="int32"),
        }
        y_full = np.random.normal(size=(num_samples, 32, 768)).astype("float32")

        start_idx = rank * base_batch_size
        indices = []
        indices.extend(range(start_idx, start_idx + base_batch_size))
        for step in range(1, 6):
            base = step * global_batch_size + start_idx
            indices.extend(range(base, base + base_batch_size))

        x = {k: v[indices] for k, v in x_full.items()}
        y = y_full[indices]

        from torch.nn.attention import sdpa_kernel

        with sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            model.fit(
                {k: v[:base_batch_size] for k, v in x.items()},
                y[:base_batch_size],
                batch_size=base_batch_size,
                epochs=1,
                steps_per_epoch=1,
                verbose=1 if rank == 0 else 0,
                shuffle=False,
            )

            if dist.is_initialized():
                dist.barrier()
            start_time = time.time()

            epochs = 1
            steps_per_epoch = 5
            history = model.fit(
                {k: v[base_batch_size:] for k, v in x.items()},
                y[base_batch_size:],
                batch_size=base_batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                verbose=1 if rank == 0 else 0,
                shuffle=False,
            )

            if dist.is_initialized():
                dist.barrier()
            end_time = time.time()

            final_loss = float(history.history["loss"][0])
            training_time = end_time - start_time
            total_samples = global_batch_size * steps_per_epoch * epochs
            throughput = total_samples / training_time
            perplexity = float(np.exp(final_loss))

    if rank == 0:
        results = {
            "final_loss": final_loss,
            "perplexity": perplexity,
            "throughput": throughput,
        }
        with open("results_torch_dp.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_backend(sys.argv[1])
    else:
        print("Running JAX backend...", flush=True)
        subprocess.run([sys.executable, __file__, "jax"], check=True)
        print("\nRunning Torch backend...", flush=True)
        subprocess.run([sys.executable, __file__, "torch"], check=True)

        try:
            with open("results_jax_dp.json", "r") as f:
                jax_res = json.load(f)
            with open("results_torch_dp.json", "r") as f:
                torch_res = json.load(f)
        except FileNotFoundError:
            print("Missing results files.")
            sys.exit(1)

        print(
            "\n"
            + f"{'Metric':<30} | {'JAX':<20} | {'Torch':<20} | {'Diff':<15}"
        )
        print("-" * 95)

        metrics = [
            ("Final Loss", "final_loss"),
            ("Perplexity", "perplexity"),
            ("Throughput (samples/sec)", "throughput"),
        ]

        for label, key in metrics:
            v_jax = jax_res[key]
            v_torch = torch_res[key]
            diff = abs(v_jax - v_torch)
            print(f"{v_jax:<20.12f} | {v_torch:<20.12f} | {diff:<15.8e}")
