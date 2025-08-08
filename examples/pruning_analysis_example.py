"""
Example: Complete Pruning Analysis with Sparsity Verification and Performance Benchmarking

This example demonstrates how to:
1. Train a simple model
2. Apply different pruning methods 
3. Verify actual sparsity achieved
4. Measure inference time improvements
"""

import keras
import numpy as np
from keras.src.pruning import PruningConfig
from keras.src.pruning import complete_pruning_analysis, analyze_sparsity, benchmark_inference
from keras.src.pruning import compare_sparsity, compare_inference_speed
from keras.src.pruning import print_sparsity_report, print_benchmark_report

# Create a simple model
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Create and train a model
    print("🚀 Creating and training model...")
    model = create_model()
    
    # Generate some dummy data
    x_train = np.random.random((1000, 784))
    y_train = np.random.randint(0, 10, (1000,))
    x_test = np.random.random((200, 784))
    y_test = np.random.randint(0, 10, (200,))
    
    # Train briefly
    model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=1, validation_split=0.2)
    
    # Prepare test data for inference benchmarking
    test_batch = x_test[:32]  # Small batch for benchmarking
    
    print("\n" + "="*80)
    print("ORIGINAL MODEL ANALYSIS")
    print("="*80)
    
    # Analyze original model sparsity
    original_stats = analyze_sparsity(model)
    print_sparsity_report(original_stats, "Original Model Sparsity")
    
    # Benchmark original model
    original_benchmark = benchmark_inference(model, test_batch, num_iterations=50)
    print_benchmark_report(original_benchmark, "Original Model Performance")
    
    print("\n" + "="*80)
    print("PRUNING WITH DIFFERENT METHODS")
    print("="*80)
    
    # Test different pruning methods
    pruning_methods = [
        ("L1", "l1"),
        ("Saliency", "saliency"), 
        ("Taylor", "taylor")
    ]
    
    dataset = (x_train[:100], y_train[:100])  # Small dataset for gradient computation
    
    for method_name, method_type in pruning_methods:
        print(f"\n🔧 Testing {method_name} Pruning...")
        
        # Create pruning config
        if method_type in ["saliency", "taylor"]:
            config = PruningConfig(
                sparsity=0.5,  # 50% sparsity
                method=method_type,
                dataset=dataset,
                loss_fn=model.loss
            )
        else:
            config = PruningConfig(
                sparsity=0.5,
                method=method_type
            )
        
        # Clone and prune model
        pruned_model = keras.models.clone_model(model)
        pruned_model.set_weights(model.get_weights())
        
        try:
            stats = pruned_model.prune(config)
            print(f"✅ {method_name} pruning completed!")
            print(f"   Target sparsity: {config.sparsity:.2f}")
            print(f"   Achieved sparsity: {stats.get('final_sparsity', 'Unknown'):.2f}")
            
            # Run complete analysis
            analysis = complete_pruning_analysis(
                model_before=model,
                model_after=pruned_model,
                test_data=test_batch,
                num_iterations=50
            )
            
            # Save results summary
            sparsity_improvement = analysis['sparsity_analysis']['changes']['sparsity_increase']
            speed_improvement = analysis['performance_analysis']['improvements']['speedup_factor']
            
            print(f"\n📋 {method_name} PRUNING SUMMARY:")
            print(f"   Sparsity achieved: {sparsity_improvement*100:.2f}% increase")
            print(f"   Speed improvement: {speed_improvement:.3f}x faster")
            print(f"   Weights pruned: {analysis['sparsity_analysis']['changes']['weights_pruned']:,}")
            
        except Exception as e:
            print(f"❌ {method_name} pruning failed: {e}")
    
    print("\n" + "="*80)
    print("DETAILED LAYER-BY-LAYER ANALYSIS")
    print("="*80)
    
    # Detailed analysis for L1 pruning (most reliable)
    print("\n🔍 Detailed L1 Pruning Analysis...")
    
    l1_config = PruningConfig(sparsity=0.3, method="l1")  # 30% sparsity
    detailed_model = keras.models.clone_model(model)
    detailed_model.set_weights(model.get_weights())
    
    # Analyze before pruning
    before_analysis = analyze_sparsity(detailed_model)
    
    # Apply pruning
    detailed_model.prune(l1_config)
    
    # Analyze after pruning
    after_analysis = analyze_sparsity(detailed_model)
    
    # Compare layer by layer
    comparison = compare_sparsity(model, detailed_model)
    print_sparsity_report(comparison, "Detailed Layer-by-Layer Comparison")
    
    # Test inference on different batch sizes
    print("\n⚡ Inference Speed vs Batch Size:")
    batch_sizes = [1, 8, 32, 64]
    
    for batch_size in batch_sizes:
        if batch_size <= len(x_test):
            test_data = x_test[:batch_size]
            
            # Benchmark original
            orig_time = benchmark_inference(model, test_data, num_iterations=30, warmup_iterations=5)
            
            # Benchmark pruned
            pruned_time = benchmark_inference(detailed_model, test_data, num_iterations=30, warmup_iterations=5)
            
            speedup = orig_time['mean_time'] / pruned_time['mean_time']
            
            print(f"   Batch size {batch_size:2d}: "
                  f"Original={orig_time['mean_time']*1000:6.2f}ms, "
                  f"Pruned={pruned_time['mean_time']*1000:6.2f}ms, "
                  f"Speedup={speedup:.3f}x")
    
    print(f"\n🎉 Analysis complete! Check the detailed reports above.")

if __name__ == "__main__":
    main()
