"""Utility functions for pruning analysis and verification."""

import time
import numpy as np
from keras.src import ops
from keras.src import backend
from keras.src.api_export import keras_export


@keras_export("keras.pruning.analyze_sparsity")
def analyze_sparsity(model, layer_names=None, tolerance=1e-8):
    """Analyze sparsity statistics for a model.
    
    Args:
        model: Keras model to analyze.
        layer_names: List of layer names to analyze, regex patterns, or None.
            - None: Analyzes all layers with weights (default)
            - List of strings: Can be exact layer names or regex patterns
            - Single string: Treated as layer name or regex pattern
        tolerance: Threshold below which weights are considered zero.
    
    Returns:
        Dictionary with sparsity statistics:
        - 'overall_sparsity': Overall sparsity across all analyzed layers
        - 'layer_stats': Per-layer statistics
        - 'total_weights': Total number of weights
        - 'zero_weights': Total number of zero weights
    """
    from keras.src.pruning.core import match_layers_by_patterns
    
    layer_stats = {}
    total_weights = 0
    total_zero_weights = 0
    
    layers_to_analyze = []
    if layer_names is None:
        # Analyze all layers with kernel weights
        layers_to_analyze = [layer for layer in model.layers 
                           if hasattr(layer, 'kernel') and layer.kernel is not None]
    else:
        # Use pattern matching to find layers
        matched_layer_names = match_layers_by_patterns(model, layer_names)
        layer_dict = {layer.name: layer for layer in model.layers}
        layers_to_analyze = [layer_dict[name] for name in matched_layer_names 
                           if name in layer_dict and hasattr(layer_dict[name], 'kernel')]
    
    for layer in layers_to_analyze:
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            weights = layer.kernel
            weights_np = backend.convert_to_numpy(weights)
            
            # Count total and zero weights
            layer_total = weights_np.size
            layer_zeros = np.sum(np.abs(weights_np) <= tolerance)
            layer_nonzeros = layer_total - layer_zeros
            layer_sparsity = layer_zeros / layer_total if layer_total > 0 else 0.0
            
            layer_stats[layer.name] = {
                'total_weights': layer_total,
                'zero_weights': layer_zeros,
                'nonzero_weights': layer_nonzeros,
                'sparsity': layer_sparsity,
                'density': 1.0 - layer_sparsity,
                'weight_shape': weights_np.shape
            }
            
            total_weights += layer_total
            total_zero_weights += layer_zeros
    
    overall_sparsity = total_zero_weights / total_weights if total_weights > 0 else 0.0
    
    return {
        'overall_sparsity': overall_sparsity,
        'overall_density': 1.0 - overall_sparsity,
        'layer_stats': layer_stats,
        'total_weights': total_weights,
        'zero_weights': total_zero_weights,
        'nonzero_weights': total_weights - total_zero_weights,
        'layers_analyzed': [layer.name for layer in layers_to_analyze],
        'layer_filter': layer_names
    }


@keras_export("keras.pruning.compare_sparsity")
def compare_sparsity(model_before, model_after, layer_names=None, tolerance=1e-8):
    """Compare sparsity between two models (before and after pruning).
    
    Args:
        model_before: Model before pruning.
        model_after: Model after pruning.
        layer_names: List of layer names to compare. If None, compares all layers.
        tolerance: Threshold below which weights are considered zero.
    
    Returns:
        Dictionary with comparison statistics.
    """
    stats_before = analyze_sparsity(model_before, layer_names, tolerance)
    stats_after = analyze_sparsity(model_after, layer_names, tolerance)
    
    comparison = {
        'before': stats_before,
        'after': stats_after,
        'changes': {
            'sparsity_increase': stats_after['overall_sparsity'] - stats_before['overall_sparsity'],
            'weights_pruned': stats_after['zero_weights'] - stats_before['zero_weights'],
            'weights_remaining': stats_after['nonzero_weights']
        }
    }
    
    # Per-layer comparison
    layer_comparisons = {}
    for layer_name in stats_before['layer_stats']:
        if layer_name in stats_after['layer_stats']:
            before_layer = stats_before['layer_stats'][layer_name]
            after_layer = stats_after['layer_stats'][layer_name]
            
            layer_comparisons[layer_name] = {
                'sparsity_before': before_layer['sparsity'],
                'sparsity_after': after_layer['sparsity'],
                'sparsity_increase': after_layer['sparsity'] - before_layer['sparsity'],
                'weights_pruned': after_layer['zero_weights'] - before_layer['zero_weights'],
                'weights_remaining': after_layer['nonzero_weights']
            }
    
    comparison['layer_comparisons'] = layer_comparisons
    return comparison


@keras_export("keras.pruning.print_sparsity_report")
def print_sparsity_report(sparsity_stats, title="Model Sparsity Analysis"):
    """Print a formatted sparsity report.
    
    Args:
        sparsity_stats: Output from analyze_sparsity() or compare_sparsity().
        title: Title for the report.
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    if 'before' in sparsity_stats and 'after' in sparsity_stats:
        # This is a comparison report
        before = sparsity_stats['before']
        after = sparsity_stats['after']
        changes = sparsity_stats['changes']
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Before pruning:")
        print(f"    Total weights: {before['total_weights']:,}")
        print(f"    Zero weights:  {before['zero_weights']:,}")
        print(f"    Sparsity:      {before['overall_sparsity']:.4f} ({before['overall_sparsity']*100:.2f}%)")
        
        print(f"\n  After pruning:")
        print(f"    Total weights: {after['total_weights']:,}")
        print(f"    Zero weights:  {after['zero_weights']:,}")
        print(f"    Sparsity:      {after['overall_sparsity']:.4f} ({after['overall_sparsity']*100:.2f}%)")
        
        print(f"\n  Changes:")
        print(f"    Weights pruned:    {changes['weights_pruned']:,}")
        print(f"    Weights remaining: {changes['weights_remaining']:,}")
        print(f"    Sparsity increase: {changes['sparsity_increase']:.4f} ({changes['sparsity_increase']*100:.2f}%)")
        
        print(f"\nPER-LAYER COMPARISON:")
        print(f"{'Layer':<25} {'Before':<12} {'After':<12} {'Pruned':<12} {'Increase':<12}")
        print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        
        for layer_name, layer_comp in sparsity_stats['layer_comparisons'].items():
            print(f"{layer_name:<25} "
                  f"{layer_comp['sparsity_before']*100:>8.2f}%   "
                  f"{layer_comp['sparsity_after']*100:>8.2f}%   "
                  f"{layer_comp['weights_pruned']:>8,}    "
                  f"{layer_comp['sparsity_increase']*100:>8.2f}%")
                  
    else:
        # This is a single model report
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total weights: {sparsity_stats['total_weights']:,}")
        print(f"  Zero weights:  {sparsity_stats['zero_weights']:,}")
        print(f"  Nonzero weights: {sparsity_stats['nonzero_weights']:,}")
        print(f"  Overall sparsity: {sparsity_stats['overall_sparsity']:.4f} ({sparsity_stats['overall_sparsity']*100:.2f}%)")
        print(f"  Overall density:  {sparsity_stats['overall_density']:.4f} ({sparsity_stats['overall_density']*100:.2f}%)")
        
        print(f"\nPER-LAYER STATISTICS:")
        print(f"{'Layer':<25} {'Shape':<20} {'Total':<12} {'Zeros':<12} {'Sparsity':<12}")
        print(f"{'-'*25} {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
        
        for layer_name, layer_stats in sparsity_stats['layer_stats'].items():
            shape_str = str(layer_stats['weight_shape'])
            print(f"{layer_name:<25} "
                  f"{shape_str:<20} "
                  f"{layer_stats['total_weights']:>8,}    "
                  f"{layer_stats['zero_weights']:>8,}    "
                  f"{layer_stats['sparsity']*100:>8.2f}%")
    
    print(f"{'='*60}\n")


@keras_export("keras.pruning.benchmark_inference")
def benchmark_inference(model, test_data, num_iterations=100, warmup_iterations=10):
    """Benchmark inference time for a model.
    
    Args:
        model: Keras model to benchmark.
        test_data: Input data for inference (numpy array or tensor).
        num_iterations: Number of inference iterations to run.
        warmup_iterations: Number of warmup iterations (not counted in timing).
    
    Returns:
        Dictionary with timing statistics:
        - 'mean_time': Mean inference time per iteration
        - 'std_time': Standard deviation of inference times
        - 'min_time': Minimum inference time
        - 'max_time': Maximum inference time
        - 'total_time': Total time for all iterations
        - 'throughput': Samples per second (if batch size > 1)
    """
    # Convert to tensor if needed
    if not hasattr(test_data, 'shape'):
        test_data = ops.convert_to_tensor(test_data)
    
    batch_size = test_data.shape[0] if len(test_data.shape) > 1 else 1
    
    # Warmup iterations
    print(f"Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        _ = model(test_data, training=False)
    
    # Actual benchmark iterations
    print(f"Running {num_iterations} benchmark iterations...")
    times = []
    
    for i in range(num_iterations):
        start_time = time.perf_counter()
        _ = model(test_data, training=False)
        end_time = time.perf_counter()
        
        iteration_time = end_time - start_time
        times.append(iteration_time)
        
        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{num_iterations} iterations...")
    
    times = np.array(times)
    
    results = {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'total_time': np.sum(times),
        'median_time': np.median(times),
        'iterations': num_iterations,
        'batch_size': batch_size
    }
    
    if batch_size > 1:
        results['throughput_samples_per_sec'] = batch_size / results['mean_time']
        results['throughput_batches_per_sec'] = 1.0 / results['mean_time']
    
    return results


@keras_export("keras.pruning.compare_inference_speed")
def compare_inference_speed(model_before, model_after, test_data, 
                           num_iterations=100, warmup_iterations=10):
    """Compare inference speed between two models.
    
    Args:
        model_before: Original model (before pruning).
        model_after: Pruned model (after pruning).
        test_data: Input data for inference.
        num_iterations: Number of iterations for benchmarking.
        warmup_iterations: Number of warmup iterations.
    
    Returns:
        Dictionary with comparison results.
    """
    print("Benchmarking original model...")
    before_stats = benchmark_inference(model_before, test_data, num_iterations, warmup_iterations)
    
    print("\nBenchmarking pruned model...")
    after_stats = benchmark_inference(model_after, test_data, num_iterations, warmup_iterations)
    
    # Calculate improvements
    speedup = before_stats['mean_time'] / after_stats['mean_time']
    time_reduction = (before_stats['mean_time'] - after_stats['mean_time']) / before_stats['mean_time']
    
    comparison = {
        'before': before_stats,
        'after': after_stats,
        'improvements': {
            'speedup_factor': speedup,
            'time_reduction_percent': time_reduction * 100,
            'time_saved_ms': (before_stats['mean_time'] - after_stats['mean_time']) * 1000
        }
    }
    
    if 'throughput_samples_per_sec' in before_stats:
        throughput_improvement = after_stats['throughput_samples_per_sec'] / before_stats['throughput_samples_per_sec']
        comparison['improvements']['throughput_improvement'] = throughput_improvement
    
    return comparison


@keras_export("keras.pruning.print_benchmark_report")
def print_benchmark_report(benchmark_stats, title="Inference Benchmark Results"):
    """Print a formatted benchmark report.
    
    Args:
        benchmark_stats: Output from benchmark_inference() or compare_inference_speed().
        title: Title for the report.
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    if 'before' in benchmark_stats and 'after' in benchmark_stats:
        # This is a comparison report
        before = benchmark_stats['before']
        after = benchmark_stats['after']
        improvements = benchmark_stats['improvements']
        
        print(f"\nTIMING COMPARISON:")
        print(f"  Original model:")
        print(f"    Mean time:   {before['mean_time']*1000:.3f} ms")
        print(f"    Std time:    {before['std_time']*1000:.3f} ms")
        print(f"    Min time:    {before['min_time']*1000:.3f} ms")
        print(f"    Max time:    {before['max_time']*1000:.3f} ms")
        
        print(f"\n  Pruned model:")
        print(f"    Mean time:   {after['mean_time']*1000:.3f} ms")
        print(f"    Std time:    {after['std_time']*1000:.3f} ms")
        print(f"    Min time:    {after['min_time']*1000:.3f} ms")
        print(f"    Max time:    {after['max_time']*1000:.3f} ms")
        
        print(f"\n  IMPROVEMENTS:")
        print(f"    Speedup factor:      {improvements['speedup_factor']:.3f}x")
        print(f"    Time reduction:      {improvements['time_reduction_percent']:.2f}%")
        print(f"    Time saved per run:  {improvements['time_saved_ms']:.3f} ms")
        
        if 'throughput_improvement' in improvements:
            print(f"    Throughput improvement: {improvements['throughput_improvement']:.3f}x")
            print(f"    Before throughput: {before['throughput_samples_per_sec']:.1f} samples/sec")
            print(f"    After throughput:  {after['throughput_samples_per_sec']:.1f} samples/sec")
            
    else:
        # Single model report
        print(f"\nTIMING STATISTICS:")
        print(f"  Iterations:  {benchmark_stats['iterations']}")
        print(f"  Batch size:  {benchmark_stats['batch_size']}")
        print(f"  Mean time:   {benchmark_stats['mean_time']*1000:.3f} ms")
        print(f"  Std time:    {benchmark_stats['std_time']*1000:.3f} ms")
        print(f"  Min time:    {benchmark_stats['min_time']*1000:.3f} ms")
        print(f"  Max time:    {benchmark_stats['max_time']*1000:.3f} ms")
        print(f"  Median time: {benchmark_stats['median_time']*1000:.3f} ms")
        
        if 'throughput_samples_per_sec' in benchmark_stats:
            print(f"  Throughput:  {benchmark_stats['throughput_samples_per_sec']:.1f} samples/sec")
    
    print(f"{'='*60}\n")


# Convenience function to run complete analysis
@keras_export("keras.pruning.complete_pruning_analysis")
def complete_pruning_analysis(model_before, model_after, test_data, 
                            layer_names=None, num_iterations=100):
    """Run complete analysis comparing models before and after pruning.
    
    Args:
        model_before: Original model.
        model_after: Pruned model.
        test_data: Test data for inference benchmarking.
        layer_names: Specific layers to analyze (None for all).
        num_iterations: Number of benchmark iterations.
    
    Returns:
        Dictionary with both sparsity and performance analysis.
    """
    print("🔍 Running complete pruning analysis...")
    
    # Sparsity analysis
    print("\n📊 Analyzing sparsity...")
    sparsity_comparison = compare_sparsity(model_before, model_after, layer_names)
    print_sparsity_report(sparsity_comparison, "Sparsity Analysis: Before vs After Pruning")
    
    # Performance benchmark
    print("\n⚡ Benchmarking inference performance...")
    speed_comparison = compare_inference_speed(model_before, model_after, test_data, num_iterations)
    print_benchmark_report(speed_comparison, "Performance Benchmark: Before vs After Pruning")
    
    return {
        'sparsity_analysis': sparsity_comparison,
        'performance_analysis': speed_comparison
    }
