import json
import math


def compare():
    try:
        with open("results_jax.json", "r") as f:
            jax = json.load(f)
        with open("results_torch.json", "r") as f:
            torch = json.load(f)
    except FileNotFoundError:
        print("Missing results files.")
        return

    print(f"{'Metric':<30} | {'JAX':<20} | {'Torch':<20} | {'Diff':<15}")
    print("-" * 95)

    metrics = [
        ("Final Loss", "final_loss"),
        ("Perplexity", "perplexity"),
        ("Throughput (samples/sec)", "throughput"),
        ("Training Time (sec)", "training_time"),
        ("Peak Memory (MB)", "peak_memory_mb"),
    ]

    for label, key in metrics:
        v_jax = jax.get(key, float("nan"))
        v_torch = torch.get(key, float("nan"))

        is_nan = math.isnan(v_jax) or math.isnan(v_torch)
        diff = abs(v_jax - v_torch) if not is_nan else float("nan")

        print(
            f"""{label:<30} | {v_jax:<20.12f} | {v_torch:<20.12f} | 
            {diff:<15.8e}"""
        )


if __name__ == "__main__":
    compare()
