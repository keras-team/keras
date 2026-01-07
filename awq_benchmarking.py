# -*- coding: utf-8 -*-
"""native_awq_benchmarking.py

AWQ (Activation-aware Weight Quantization) benchmarking script.
Adapted from GPTQ benchmarking script.
"""

# Uncomment below for Colab setup:
# !pip uninstall -y -q keras keras-hub
# !pip install -q git+https://github.com/JyotinderSingh/keras@awq-2
# !pip install -q git+https://github.com/keras-team/keras-hub
# !pip install -q -U "datasets>=2.19" "fsspec>=2024.3" "huggingface_hub>=0.23"
# psutil nvidia-ml-py

import csv
import gc
import inspect
import io
import logging
import os
import shutil
import tarfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import keras_hub
import numpy as np
import requests
import tensorflow as tf
from datasets import load_dataset
from tqdm import tqdm

import keras
from keras import losses
from keras import ops
from keras.quantizers import AWQConfig

# ---------------------------
# Logging
# ---------------------------


def setup_logging(level=logging.ERROR):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


setup_logging()

# ---------------------------
# Utilities: humanize bytes
# ---------------------------


def human_bytes(n: int | None) -> str:
    if n is None:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} EB"


# ---------------------------
# Dataset helpers
# ---------------------------


def get_dataset_text(dataset_name: str, split: str = "train") -> str:
    """
    Download and return text for small test corpora.
    """
    logging.info("Loading dataset '%s' split='%s'...", dataset_name, split)

    if dataset_name == "wikitext2":
        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        return "\n\n".join(d["text"] for d in raw_dataset)

    if dataset_name == "ptb":
        url = "https://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            file_path = f"./simple-examples/data/ptb.{split}.txt"
            text_bytes = tar.extractfile(file_path).read()
        return text_bytes.decode("utf-8")

    raise ValueError(f"Unsupported dataset name for testing: {dataset_name!r}")


def build_token_dataloader(
    all_tokens: np.ndarray, seq_len: int, max_batches: int = 50
) -> np.ndarray:
    """
    Slice a long token stream into [B, T] windows for perplexity evaluation.
    """
    samples = []
    for i in range(max_batches):
        start = i * seq_len
        end = start + seq_len
        if end > len(all_tokens):
            break
        samples.append(np.reshape(all_tokens[start:end], (1, seq_len)))
    if not samples:
        raise ValueError(
            "Not enough tokens to build evaluation batches. "
            f"Need >= {seq_len}, got {len(all_tokens)}."
        )
    return np.array(samples, dtype=np.int32)


# ---------------------------
# Instrumentation: CPU/GPU memory + time
# ---------------------------

# psutil is optional; degrade gracefully
try:
    import psutil

    _PSUTIL_OK = True
except Exception:
    _PSUTIL_OK = False
    psutil = None  # type: ignore


def _gpu_devices():
    try:
        return tf.config.list_physical_devices("GPU")
    except Exception:
        return []


def _gpu_mem_supported() -> bool:
    # Requires TF 2.9+ (get_memory_info/reset_memory_stats)
    return hasattr(tf.config.experimental, "get_memory_info") and hasattr(
        tf.config.experimental, "reset_memory_stats"
    )


def gpu_reset_peaks():
    if not _gpu_mem_supported():
        return
    for i, _ in enumerate(_gpu_devices()):
        tf.config.experimental.reset_memory_stats(f"GPU:{i}")


def gpu_peaks() -> dict[int, dict[str, int]]:
    """
    Returns {gpu_index: {'current': bytes, 'peak': bytes}} since last reset.
    """
    out: dict[int, dict[str, int]] = {}
    if not _gpu_mem_supported():
        return out
    for i, _ in enumerate(_gpu_devices()):
        info = tf.config.experimental.get_memory_info(
            f"GPU:{i}"
        )  # {'current','peak'}
        out[i] = {
            "current": int(info.get("current", 0)),
            "peak": int(info.get("peak", 0)),
        }
    return out


class CPUMemSampler:
    """
    Poll process RSS while running to estimate per-window peak main memory.
    """

    def __init__(self, interval_sec: float = 0.05):
        self.interval = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak = 0
        self._proc = psutil.Process(os.getpid()) if _PSUTIL_OK else None

    def start(self):
        if self._proc is None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._proc is None:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss  # bytes
            if rss > self._peak:
                self._peak = rss
            time.sleep(self.interval)

    @property
    def peak_bytes(self) -> int | None:
        return self._peak if self._proc is not None else None


@contextmanager
def profile_quantization():
    """
    Context manager that captures:
      - wall time
      - CPU RSS peak during window
      - per-GPU current + peak bytes during window

    Usage:
        with profile_quantization() as prof:
            model.quantize("awq", config=cfg)
        logging.info(prof.summary())
    """
    # CPU sampling
    cpu = CPUMemSampler(interval_sec=0.05)
    cpu.start()

    # GPU: baseline + reset peaks for a clean window
    have_gpu = len(_gpu_devices()) > 0 and _gpu_mem_supported()
    baseline_gpu = gpu_peaks() if have_gpu else {}

    if have_gpu:
        gpu_reset_peaks()

    t0 = time.perf_counter()
    results = {
        "elapsed_sec": None,
        "cpu_peak_bytes": None,
        "gpu_stats": {},
        "gpu_baseline": baseline_gpu,
    }

    try:
        yield results
    finally:
        elapsed = time.perf_counter() - t0
        cpu.stop()
        results["elapsed_sec"] = elapsed
        results["cpu_peak_bytes"] = cpu.peak_bytes
        results["gpu_stats"] = gpu_peaks() if have_gpu else {}


def summarize_profile(results: dict) -> str:
    lines = []
    lines.append(f"Quantization time: {results['elapsed_sec']:.3f} s")
    lines.append(
        f"CPU peak RSS (window): {human_bytes(results['cpu_peak_bytes'])}"
    )
    gpu_stats = results.get("gpu_stats") or {}
    if gpu_stats:
        for i, d in gpu_stats.items():
            cur, peak = d.get("current", 0), d.get("peak", 0)
            lines.append(
                f"GPU:{i} current: {human_bytes(cur)} | peak (window): {human_bytes(peak)}"  # noqa: E501
            )
    else:
        if len(_gpu_devices()) == 0:
            lines.append("GPU metrics: no GPU detected")
        elif not _gpu_mem_supported():
            lines.append(
                "GPU metrics: TF build lacks get_memory_info/reset_memory_stats"
            )
    return "\n".join(lines)


def reset_resources():
    """
    Clear Python, Keras, and TF state to avoid memory carry-over
    between benchmarks.
    """
    logging.info("Resetting resources before benchmark...")
    try:
        keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()
    if _gpu_mem_supported():
        gpu_reset_peaks()
    logging.info("Resources reset complete.")


# ---------------------------
# Model saving & on-disk size
# ---------------------------


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    # directory: sum all files
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _safe_remove(path: Path):
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.is_file():
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def save_model_and_size(model, filename_no_ext: str) -> int:
    """
    Saves model to disk using model.save and returns on-disk size in bytes.
    Uses Keras v3 .keras single-file format by default.
    """
    out_path = Path(f"{filename_no_ext}.keras")
    _safe_remove(out_path)
    model.save(str(out_path))
    size_bytes = _path_size_bytes(out_path)
    logging.info("Saved model to %s (%s).", out_path, human_bytes(size_bytes))
    return size_bytes


# ---------------------------
# Perplexity evaluation
# ---------------------------


def calculate_perplexity(model, dataloader: np.ndarray) -> float:
    """
    Compute perplexity on a token dataloader: [B, T] int32.
    Compatible with Keras ops; backend-agnostic.
    """
    logging.info("Evaluating perplexity on %d batches...", len(dataloader))
    total_nll = ops.zeros((), dtype="float32")
    total_tokens = ops.zeros((), dtype="float32")

    # Mask uses token id != 1 (commonly EOS in some presets). Adjust if needed.
    for batch in tqdm(dataloader, desc="PPL", leave=False):
        batch = ops.convert_to_tensor(batch, dtype="int32")
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        # If model has a preprocessor, pass the structured dict; else raw ids
        if hasattr(model, "preprocessor") and model.preprocessor is not None:
            inputs = {
                "token_ids": input_ids,
                "padding_mask": ops.ones_like(input_ids, dtype="bool"),
            }
        else:
            inputs = input_ids

        outputs = model(inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        loss_fn = losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=None
        )
        token_loss = loss_fn(ops.expand_dims(targets, -1), logits)
        mask = ops.cast(ops.not_equal(targets, 1), dtype="float32")
        masked = token_loss * mask

        total_nll = total_nll + ops.sum(masked)
        total_tokens = total_tokens + ops.sum(mask)

    # Guard against zero tokens (e.g., degenerate slicing)
    if float(total_tokens) == 0.0:
        logging.warning("No tokens were evaluated; returning perplexity=inf.")
        return float("inf")

    ppl = ops.exp(total_nll / total_tokens)
    ppl_value = float(ppl)
    logging.info("Perplexity: %.4f", ppl_value)
    return ppl_value


# ---------------------------
# Generation utilities + benchmarks (first-token latency & throughput)
# ---------------------------


def _supports_arg(fn, argname: str) -> bool:
    try:
        sig = inspect.signature(fn)
        return argname in sig.parameters
    except Exception:
        return False


def _extract_texts_from_generate_output(output):
    """
    Try to normalize model.generate outputs into a list[str].
    """
    if output is None:
        return []
    # Common patterns: list[str], dict with "text" (list[str] or str),
    # list[dict{"text": str}]
    if isinstance(output, list):
        if all(isinstance(x, str) for x in output):
            return output
        if all(isinstance(x, dict) and "text" in x for x in output):
            return [x["text"] for x in output]
    if isinstance(output, dict):
        val = output.get("text")
        if isinstance(val, str):
            return [val]
        if isinstance(val, list) and all(isinstance(x, str) for x in val):
            return val
    # Fallback: best effort string conversion
    return [str(output)]


def _token_count(model, text: str) -> int:
    try:
        tok = model.preprocessor.tokenizer.tokenize(text)
        return int(len(tok))
    except Exception:
        # Unknown tokenizer; fallback to rough word count
        return int(len(text.split()))


def _prompt_token_len(model, prompt: str) -> int:
    return _token_count(model, prompt)


def _generate(
    model,
    prompts: list[str],
    *,
    max_new_tokens: int | None = None,
    max_length: int | None = None,
):
    """
    Thin wrapper that tries (max_new_tokens) first, then (max_length).
    Expects models with a .generate method that accept {"prompts": [...]} input.
    """
    if hasattr(model, "generate"):
        try:
            if max_new_tokens is not None and _supports_arg(
                model.generate, "max_new_tokens"
            ):
                return model.generate(prompts, max_new_tokens=max_new_tokens)
        except TypeError:
            pass
        # Fallback to max_length
        if max_length is not None:
            return model.generate(prompts, max_length=max_length)
        # Last resort: call with only prompts
        return model.generate(prompts)
    raise AttributeError("Model has no .generate(...) method")


def benchmark_first_token_latency_ms(model, prompt: str) -> float:
    """
    Approximates first-token latency by measuring a single-token generation call
    """
    try:
        try:
            _ = _generate(model, [prompt], max_new_tokens=1)
        except Exception:
            pass
        t0 = time.perf_counter()
        # Prefer max_new_tokens=1 if supported; else approximate via max_length
        if hasattr(model, "generate") and _supports_arg(
            model.generate, "max_new_tokens"
        ):
            _ = _generate(model, [prompt], max_new_tokens=1)
        else:
            base_len = _prompt_token_len(model, prompt)
            _ = _generate(model, [prompt], max_length=base_len + 1)
        t1 = time.perf_counter()
        return (t1 - t0) * 1e3
    except Exception as e:
        logging.warning("First-token latency benchmark failed: %s", e)
        return 0.0


def benchmark_generation_throughput(
    model, prompts: list[str], target_new_tokens: int = 50
) -> float:
    """
    Measures tokens/sec over a short generation run across the provided prompts.
    Returns overall throughput in tokens/sec (sum of new tokens / wall time).
    """
    try:
        # Warmup (small)
        try:
            _ = _generate(model, prompts[:1], max_new_tokens=1)
        except Exception:
            pass

        t0 = time.perf_counter()
        outputs = _generate(model, prompts, max_new_tokens=target_new_tokens)
        t1 = time.perf_counter()
        elapsed = max(t1 - t0, 1e-6)

        outs = _extract_texts_from_generate_output(outputs)
        # Pad if necessary
        while len(outs) < len(prompts):
            outs.append(outs[-1] if outs else "")

        new_tokens = 0
        for p, o in zip(prompts, outs):
            new_tokens += max(
                _token_count(model, o) - _token_count(model, p), 0
            )

        return float(new_tokens) / float(elapsed)
    except Exception as e:
        logging.warning("Throughput benchmark failed: %s", e)
        return 0.0


# ---------------------------
# Peak GPU memory used during inference (multi-backend)
# ---------------------------


def benchmark_peak_gpu_memory(model, prompts, max_length, model_name):
    """
    Measures peak GPU memory usage during a generation pass,
    dynamically adapting to the Keras backend (TensorFlow, PyTorch, or JAX).
    Returns MiB (float).
    """
    print(f"\nBenchmarking Peak Memory: {model_name}")

    backend = (
        os.environ.get("KERAS_BACKEND")
        if os.environ.get("KERAS_BACKEND")
        else "tensorflow"
    )
    peak_mem_mib = 0

    # Define the function to be profiled
    def generation_func():
        try:
            _ = model.generate(prompts, max_length=max_length)
        except TypeError:
            try:
                _ = model.generate(
                    prompts, max_new_tokens=max(1, max_length or 1)
                )
            except Exception:
                _ = model.generate(prompts)

    # --- TensorFlow Backend ---
    if backend == "tensorflow":
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                print("GPU not available for TensorFlow.")
                return 0.0

            # Reset TF's internal peak stats if available
            if hasattr(tf.config.experimental, "reset_memory_stats"):
                tf.config.experimental.reset_memory_stats("GPU:0")

            generation_func()

            # Get peak memory usage in bytes and convert to MiB
            mem_info = tf.config.experimental.get_memory_info("GPU:0")
            peak_mem_bytes = mem_info.get("peak", 0)
            peak_mem_mib = peak_mem_bytes / (1024**2)

        except ImportError:
            print("TensorFlow not found. Cannot measure GPU memory.")
            return 0.0

    # --- PyTorch Backend ---
    elif backend == "torch":
        try:
            import torch

            if not torch.cuda.is_available():
                print("GPU not available for PyTorch.")
                return 0.0

            device = torch.device("cuda:0")
            # Best effort move if the model is torch-backed
            try:
                model.to(device)  # type: ignore[attr-defined]
            except Exception:
                pass

            torch.cuda.reset_peak_memory_stats(device)
            generation_func()
            peak_mem_bytes = torch.cuda.max_memory_allocated(device)
            peak_mem_mib = peak_mem_bytes / (1024**2)

        except ImportError:
            print("PyTorch not found. Cannot measure GPU memory.")
            return 0.0

    # --- JAX Backend ---
    elif backend == "jax":
        try:
            from pynvml import nvmlDeviceGetHandleByIndex
            from pynvml import nvmlDeviceGetMemoryInfo
            from pynvml import nvmlInit
            from pynvml import nvmlShutdown

            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)  # Assumes GPU device 0

            generation_func()

            mem_info = nvmlDeviceGetMemoryInfo(handle)
            peak_mem_bytes = mem_info.used
            peak_mem_mib = peak_mem_bytes / (1024**2)
            nvmlShutdown()

        except ImportError:
            print(
                "pynvml not found. Please run 'pip install nvidia-ml-py' "
                "to measure GPU memory for the JAX backend."
            )
            return 0.0
        except Exception as e:
            print(f"An error occurred with NVML: {e}")
            return 0.0
    else:
        print(
            f"Unknown KERAS_BACKEND='{backend}'. Skipping GPU memory benchmark."
        )
        return 0.0

    return float(peak_mem_mib)


# ---------------------------
# CSV logging
# ---------------------------

CSV_HEADER = [
    "timestamp",
    "model_name",
    "model_preset",
    "dataset_name",
    "seq_len",
    "eval_batches",
    "calib_samples",
    "n_grid",
    "pre_perplexity",
    "post_perplexity",
    "quant_time_sec",
    "quant_cpu_peak_bytes",
    "quant_gpu_peak_bytes",
    "disk_size_pre_bytes",
    "disk_size_post_bytes",
    "infer_peak_gpu_mem_bytes",
    "first_token_latency_ms",
    "throughput_tokens_per_sec",
]


def append_row_to_csv(csv_path: str, row: dict):
    path = Path(csv_path)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_HEADER})
    logging.info("Appended results to %s", path)


# ---------------------------
# Main test runner
# ---------------------------


def run_quantization_test(
    model_class,
    model_preset: str,
    *,
    dataset_name: str = "wikitext2",
    seq_len: int = 128,
    eval_batches: int = 50,
    calib_samples: int = 128,
    n_grid: int = 20,
    group_size: int = 128,
    csv_path: str = "awq_benchmarks.csv",
    save_basename_prefix: str | None = None,
):
    """
    Load model + data, evaluate perplexity before/after AWQ quantization,
    report time + CPU/GPU memory statistics, save models to disk, and
    benchmark inference GPU memory, first-token latency, and throughput.

    Args:
        model_class: e.g., keras_hub.models.GPT2CausalLM
        model_preset: model preset string, e.g., "opt_125m_en"
        dataset_name: "wikitext2" or "ptb"
        seq_len: evaluation/calibration sequence length
        eval_batches: max number of [1, T] sequences for perplexity eval
        calib_samples: number of calibration snippets
        n_grid: number of grid search points for AWQ optimal scale finding
        group_size: weight group size for quantization
        csv_path: file to append benchmark rows
        save_basename_prefix: optional prefix for saved model filenames
    """
    if AWQConfig is None:
        logging.error("AWQConfig unavailable; cannot run test.")
        return

    reset_resources()

    logging.info("========== AWQ Quantization Test ==========")
    logging.info("Model preset: %s", model_preset)
    logging.info(
        "Dataset: %s | seq_len=%d | eval_batches=%d | calib_samples=%d | n_grid=%d",  # noqa: E501
        dataset_name,
        seq_len,
        eval_batches,
        calib_samples,
        n_grid,
    )

    # 1) Load model + text, tokenize
    try:
        logging.info("Loading model...")
        model = model_class.from_preset(model_preset)

        logging.info("Loading text for eval/calibration...")
        test_text = get_dataset_text(dataset_name, split="test")
        train_text = get_dataset_text(dataset_name, split="train")

        logging.info("Tokenizing test split for eval windows...")
        all_tokens = model.preprocessor.tokenizer.tokenize(test_text)
    except Exception as e:
        logging.exception("Failed during model/data load: %s", e)
        return

    # Build dataloader for perplexity
    test_dataloader = build_token_dataloader(
        all_tokens, seq_len, max_batches=eval_batches
    )

    # 2) PPL before quantization
    logging.info("Calculating perplexity BEFORE quantization...")
    pre_ppl = calculate_perplexity(model, test_dataloader)
    logging.info("Pre-quantization perplexity: %.4f", pre_ppl)

    # Calibration dataset: simple sentence split (adjust as needed)
    calibration_dataset = [
        s.strip() + "." for s in train_text.split(".") if s.strip()
    ][:calib_samples]
    # Prompts for generation benchmarks (reuse calibration sentences)
    gen_prompts = (
        calibration_dataset if calibration_dataset else ["The quick brown fox"]
    )

    # ---------------------------
    # Save model pre-quantization & measure disk size
    # ---------------------------
    prefix = save_basename_prefix or model.name or "model"
    disk_size_pre = save_model_and_size(model, f"{prefix}_pre")

    # AWQ config
    # Note: AWQ only supports 4-bit quantization (weight_bits=4)
    awq_config = AWQConfig(
        dataset=calibration_dataset,
        tokenizer=(
            model.preprocessor.tokenizer
            if hasattr(model, "preprocessor")
            else None
        ),
        weight_bits=4,  # AWQ only supports 4-bit
        num_samples=calib_samples,
        sequence_length=seq_len,
        group_size=group_size,
        n_grid=n_grid,  # AWQ-specific: number of grid search points
    )

    # Optional: pre/post snapshots (RSS + GPU current)
    pre_cpu = (
        psutil.Process(os.getpid()).memory_info().rss if _PSUTIL_OK else None
    )
    pre_gpu = gpu_peaks() if _gpu_mem_supported() else {}

    # 3) Quantize with instrumentation
    logging.info("Quantizing with AWQ...")
    with profile_quantization() as prof:
        model.quantize("awq", config=awq_config)
    logging.info("Quantization complete.")
    logging.info("\n%s", summarize_profile(prof))

    post_cpu = (
        psutil.Process(os.getpid()).memory_info().rss if _PSUTIL_OK else None
    )
    post_gpu = gpu_peaks() if _gpu_mem_supported() else {}

    # Report snapshots
    logging.info(
        "CPU RSS pre/post: %s -> %s (Δ %s)",
        human_bytes(pre_cpu),
        human_bytes(post_cpu),
        human_bytes(
            None
            if (pre_cpu is None or post_cpu is None)
            else max(post_cpu - pre_cpu, 0)
        ),
    )
    if pre_gpu and post_gpu:
        for i in sorted(post_gpu.keys()):
            pre_cur = pre_gpu.get(i, {}).get("current", 0)
            post_cur = post_gpu.get(i, {}).get("current", 0)
            logging.info(
                "GPU:%d current pre/post: %s -> %s",
                i,
                human_bytes(pre_cur),
                human_bytes(post_cur),
            )

    # 4) PPL after quantization
    logging.info("Calculating perplexity AFTER quantization...")
    post_ppl = calculate_perplexity(model, test_dataloader)
    logging.info("Post-quantization perplexity: %.4f", post_ppl)

    # ---------------------------
    # Save model post-quantization & measure disk size
    # ---------------------------
    disk_size_post = save_model_and_size(model, f"{prefix}_awq")

    # ---------------------------
    # Inference GPU peak memory (single pass)
    # ---------------------------
    try:
        # Use a modest max length to keep runs quick and consistent
        infer_peak_mem_mib = benchmark_peak_gpu_memory(
            model=model,
            prompts=gen_prompts[:1],  # single prompt
            max_length=seq_len,  # short generation
            model_name=model.name or "model",
        )
        infer_peak_mem_bytes = int(infer_peak_mem_mib * (1024**2))
    except Exception as e:
        logging.warning("Inference GPU memory benchmark failed: %s", e)
        infer_peak_mem_bytes = 0

    # ---------------------------
    # First-token latency
    # ---------------------------
    ft_latency_ms = benchmark_first_token_latency_ms(model, gen_prompts[0])
    ft_latency_ms = benchmark_first_token_latency_ms(model, gen_prompts[0])

    # ---------------------------
    # Generation throughput
    # ---------------------------
    throughput_toks_per_sec = benchmark_generation_throughput(
        model, gen_prompts[: min(4, len(gen_prompts))], target_new_tokens=50
    )
    throughput_toks_per_sec = benchmark_generation_throughput(
        model, gen_prompts[: min(4, len(gen_prompts))], target_new_tokens=50
    )

    # Quantization GPU peak during window (aggregate max across devices)
    quant_gpu_peak = 0
    try:
        gpu_stats = (prof or {}).get("gpu_stats", {})
        if gpu_stats:
            quant_gpu_peak = max(d.get("peak", 0) for d in gpu_stats.values())
    except Exception:
        pass

    # ---------------------------
    # Write CSV row
    # ---------------------------
    csv_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model.name or "",
        "model_preset": model_preset,
        "dataset_name": dataset_name,
        "seq_len": seq_len,
        "eval_batches": eval_batches,
        "calib_samples": calib_samples,
        "n_grid": n_grid,
        "pre_perplexity": f"{pre_ppl:.6f}",
        "post_perplexity": f"{post_ppl:.6f}",
        "quant_time_sec": f"{(prof or {}).get('elapsed_sec', 0.0):.6f}",
        "quant_cpu_peak_bytes": int((prof or {}).get("cpu_peak_bytes") or 0),
        "quant_gpu_peak_bytes": int(quant_gpu_peak),
        "disk_size_pre_bytes": int(disk_size_pre),
        "disk_size_post_bytes": int(disk_size_post),
        "infer_peak_gpu_mem_bytes": int(infer_peak_mem_bytes),
        "first_token_latency_ms": f"{ft_latency_ms:.3f}",
        "throughput_tokens_per_sec": f"{throughput_toks_per_sec:.6f}",
    }
    append_row_to_csv(csv_path, csv_row)

    logging.info("============== Test Finished ==============\n")

    return model, {
        "pre_perplexity": pre_ppl,
        "post_perplexity": post_ppl,
        "profile": prof,
        "pre_cpu_bytes": pre_cpu,
        "post_cpu_bytes": post_cpu,
        "pre_gpu_stats": pre_gpu,
        "post_gpu_stats": post_gpu,
        # New returns:
        "disk_size_pre_bytes": disk_size_pre,
        "disk_size_post_bytes": disk_size_post,
        "infer_peak_gpu_mem_bytes": infer_peak_mem_bytes,
        "first_token_latency_ms": ft_latency_ms,
        "throughput_tokens_per_sec": throughput_toks_per_sec,
        "csv_path": csv_path,
    }


# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # GPT-2
    gpt2, stats = run_quantization_test(
        keras_hub.models.GPT2CausalLM, "gpt2_base_en_cnn_dailymail"
    )
    print(stats)
    del gpt2

    # # OPT-125M
    # opt125, stats = run_quantization_test(
    #     model_class=keras_hub.models.OPTCausalLM, model_preset="opt_125m_en"
    # )
    # print(stats)
    # del opt125

    # # BLOOM
    # bloom, stats = run_quantization_test(
    #     model_class=keras_hub.models.BloomCausalLM,
    #     model_preset="bloom_1.1b_multi",
    # )
    # print(stats)
    # del bloom

    # # Gemma3
    # gemma3, stats = run_quantization_test(
    #     model_class=keras_hub.models.Gemma3CausalLM, model_preset="gemma3_1b"
    # )
    # print(stats)
    # del gemma3
