#!/usr/bin/env python3
"""
Keras Performance Comparison: Local vs Pip
Compares inference latency between local optimized Keras and pip-installed baseline
"""

import time
import numpy as np
import argparse
import json
import sys
import os

def build_simple_model():
    """Build a simple CNN model"""
    import keras
    
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(10)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def build_mlp_model():
    """Build a simple MLP model"""
    import keras
    
    inputs = keras.Input(shape=(784,))
    x = keras.layers.Dense(512, activation='relu')(inputs)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(10)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def benchmark_model(model, input_shape, backend_name, num_runs=10, warmup=2):
    """Benchmark a model"""
    print(f"\nBenchmarking {backend_name}...")
    print(f"Model: {model.name} | Input shape: {input_shape}")
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Create dummy input
    x = np.random.randn(*input_shape).astype('float32')
    
    # Warmup
    print(f"Warmup ({warmup} runs)...", end="")
    for _ in range(warmup):
        _ = model.predict(x, verbose=0)
    print(" Done")
    
    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...", end="")
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model.predict(x, verbose=0)
        times.append(time.perf_counter() - t0)
    
    times = np.array(times) * 1000  # Convert to ms
    stats = {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'median': float(np.median(times)),
        'p95': float(np.percentile(times, 95)),
        'times': times.tolist()
    }
    print(" Done")
    
    print(f"  Mean: {stats['mean']:.2f}ms ± {stats['std']:.2f}ms")
    print(f"  Median: {stats['median']:.2f}ms | P95: {stats['p95']:.2f}ms")
    
    return stats

def run_comparison(backend='tensorflow'):
    """Run comparison benchmark"""
    import keras
    print(f"=" * 70)
    print(f"KERAS BACKEND: {backend}")
    print(f"=" * 70)
    
    # Test with different models
    results = {}
    
    # CNN Model
    print("\n--- CNN Model (28x28x1 input) ---")
    model_cnn = build_simple_model()
    results['cnn'] = benchmark_model(model_cnn, (1, 28, 28, 1), backend)
    
    # MLP Model
    print("\n--- MLP Model (784 input) ---")
    model_mlp = build_mlp_model()
    results['mlp'] = benchmark_model(model_mlp, (1, 784), backend)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Keras Performance Comparison')
    parser.add_argument('--backend', default='tensorflow', help='Keras backend')
    parser.add_argument('--runs', type=int, default=10, help='Number of benchmark runs')
    parser.add_argument('--output', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Set backend
    os.environ['KERAS_BACKEND'] = args.backend
    
    results = run_comparison(args.backend)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(json.dumps(results, indent=2))
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

if __name__ == '__main__':
    main()
