"""
Example: New Direct Parameter Pruning API with Layer Selection

This example demonstrates the new pruning API that:
1. Accepts parameters directly instead of config objects
2. Supports selective layer pruning using names and regex patterns
3. Provides detailed analysis of which layers were affected
"""

import keras
import numpy as np
from keras.src.pruning import complete_pruning_analysis, analyze_sparsity

def create_model():
    """Create a model with various layer types and naming patterns."""
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,), name='dense_input'),
        keras.layers.Dense(64, activation='relu', name='dense_hidden_1'),  
        keras.layers.Dense(64, activation='relu', name='dense_hidden_2'),
        keras.layers.Dense(32, activation='relu', name='dense_bottleneck'),
        keras.layers.Dense(10, activation='softmax', name='dense_output'),
        
        # Add some conv layers in a functional model for demonstration
    ])
    
    # Also create a more complex model with conv layers
    inputs = keras.Input(shape=(28, 28, 1), name='input')
    x = keras.layers.Conv2D(32, 3, activation='relu', name='conv2d_1')(inputs)
    x = keras.layers.Conv2D(64, 3, activation='relu', name='conv2d_2')(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation='relu', name='dense_features')(x)
    outputs = keras.layers.Dense(10, activation='softmax', name='dense_classifier')(x)
    
    conv_model = keras.Model(inputs=inputs, outputs=outputs, name='conv_model')
    
    return model, conv_model

def main():
    print("🚀 Creating models...")
    dense_model, conv_model = create_model()
    
    # Compile models
    dense_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    conv_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Generate dummy data
    x_dense = np.random.random((100, 784))
    y_dense = np.random.randint(0, 10, (100,))
    
    x_conv = np.random.random((100, 28, 28, 1))
    y_conv = np.random.randint(0, 10, (100,))
    
    print("\n" + "="*80)
    print("1. BASIC DIRECT PARAMETER PRUNING")
    print("="*80)
    
    # Example 1: Basic pruning with direct parameters (no config needed!)
    model1 = keras.models.clone_model(dense_model)
    model1.set_weights(dense_model.get_weights())
    
    print("\n🔧 Basic L1 pruning on all layers...")
    stats = model1.prune(sparsity=0.5, method="l1")
    
    print(f"✅ Pruning completed!")
    print(f"   Pruned {stats['pruned_layers']} layers")
    print(f"   Final sparsity: {stats['final_sparsity']:.3f}")
    print(f"   Layers pruned: {', '.join(stats['layers_pruned'])}")
    
    print("\n" + "="*80)
    print("2. SELECTIVE LAYER PRUNING BY NAME")  
    print("="*80)
    
    # Example 2: Prune only specific layers by name
    model2 = keras.models.clone_model(dense_model)
    model2.set_weights(dense_model.get_weights())
    
    layers_to_prune = ["dense_hidden_1", "dense_hidden_2"]  # Exact names
    
    print(f"\n🎯 Pruning only layers: {layers_to_prune}")
    stats = model2.prune(
        sparsity=0.6,
        method="structured",
        layers_to_prune=layers_to_prune
    )
    
    print(f"✅ Selective pruning completed!")
    print(f"   Layers specified: {stats['layers_specified']}")
    print(f"   Layers matched: {stats['layers_matched']}")
    print(f"   Layers pruned: {stats['layers_pruned']}")
    print(f"   Layers skipped: {stats['layers_skipped']}")
    
    print("\n" + "="*80)
    print("3. REGEX PATTERN LAYER SELECTION")
    print("="*80)
    
    # Example 3: Use regex patterns to select layers
    model3 = keras.models.clone_model(conv_model)
    model3.set_weights(conv_model.get_weights())
    
    regex_patterns = ["conv2d_.*", "dense_features"]  # Regex patterns
    
    print(f"\n🔍 Pruning layers matching patterns: {regex_patterns}")
    stats = model3.prune(
        sparsity=0.4,
        method="l2", 
        layers_to_prune=regex_patterns
    )
    
    print(f"✅ Pattern-based pruning completed!")
    print(f"   Patterns used: {stats['layers_specified']}")
    print(f"   Layers matched: {stats['layers_matched']}")
    print(f"   Layers pruned: {stats['layers_pruned']}")
    
    print("\n" + "="*80)
    print("4. GRADIENT-BASED PRUNING WITH DATASET")
    print("="*80)
    
    # Example 4: Saliency pruning with dataset
    model4 = keras.models.clone_model(dense_model)
    model4.set_weights(dense_model.get_weights())
    
    dataset = (x_dense[:50], y_dense[:50])  # Small sample for gradients
    
    print(f"\n🧠 Saliency pruning with gradient computation...")
    try:
        stats = model4.prune(
            sparsity=0.3,
            method="saliency",
            dataset=dataset,
            loss_fn="sparse_categorical_crossentropy",
            layers_to_prune="dense_hidden_.*"  # Single regex string
        )
        
        print(f"✅ Saliency pruning completed!")
        print(f"   Method: {stats['method']}")
        print(f"   Layers pruned: {stats['layers_pruned']}")
        print(f"   Final sparsity: {stats['final_sparsity']:.3f}")
        
    except Exception as e:
        print(f"❌ Saliency pruning failed: {e}")
        print("   (This is expected if not using TensorFlow backend)")
    
    print("\n" + "="*80)
    print("5. CALLBACK-BASED TRAINING WITH SELECTIVE PRUNING")
    print("="*80)
    
    # Example 5: Use callbacks with new parameter interface
    print(f"\n📚 Training with gradual pruning callback...")
    
    model5 = keras.models.clone_model(dense_model)
    
    # New callback interface - no config needed!
    pruning_callback = keras.callbacks.PruningCallback(
        sparsity=0.7,
        method="l1",
        start_step=10,
        end_step=50,
        frequency=10,
        layers_to_prune=["dense_hidden_.*", "dense_bottleneck"],  # Mixed patterns
        verbose=True
    )
    
    print("Training model with selective pruning...")
    model5.fit(
        x_dense, y_dense,
        epochs=2,
        batch_size=20,
        callbacks=[pruning_callback],
        verbose=0
    )
    
    print("\n" + "="*80)
    print("6. DETAILED ANALYSIS WITH LAYER FILTERING")
    print("="*80)
    
    # Example 6: Analyze sparsity of specific layer groups
    print(f"\n📊 Analyzing sparsity by layer groups...")
    
    # Analyze all layers
    all_stats = analyze_sparsity(model5)
    print(f"All layers - Total sparsity: {all_stats['overall_sparsity']:.3f}")
    print(f"Layers analyzed: {len(all_stats['layers_analyzed'])}")
    
    # Analyze only hidden layers using regex
    hidden_stats = analyze_sparsity(model5, layer_names=["dense_hidden_.*"])
    print(f"Hidden layers only - Sparsity: {hidden_stats['overall_sparsity']:.3f}")
    print(f"Hidden layers: {hidden_stats['layers_analyzed']}")
    
    # Analyze specific layers by name
    specific_stats = analyze_sparsity(model5, layer_names=["dense_input", "dense_output"])
    print(f"Input/Output layers - Sparsity: {specific_stats['overall_sparsity']:.3f}")
    print(f"Specific layers: {specific_stats['layers_analyzed']}")
    
    print("\n" + "="*80)
    print("7. COMPARISON WITH LAYER FILTERING") 
    print("="*80)
    
    # Create comparison model
    model_orig = keras.models.clone_model(dense_model)
    model_orig.set_weights(dense_model.get_weights())
    
    model_pruned = keras.models.clone_model(dense_model)
    model_pruned.set_weights(dense_model.get_weights())
    model_pruned.prune(sparsity=0.5, method="l1", layers_to_prune=["dense_hidden_.*"])
    
    # Compare with layer filtering
    print(f"\n🔍 Full model analysis...")
    analysis_full = complete_pruning_analysis(
        model_before=model_orig,
        model_after=model_pruned,
        test_data=x_dense[:20],
        num_iterations=30
    )
    
    print(f"\n🎯 Hidden layers only analysis...")
    from keras.src.pruning import compare_sparsity, print_sparsity_report
    
    hidden_comparison = compare_sparsity(
        model_orig, model_pruned, 
        layer_names=["dense_hidden_.*"]  # Only analyze hidden layers
    )
    print_sparsity_report(hidden_comparison, "Hidden Layers Comparison")
    
    print(f"\n🎉 All examples completed! Key improvements:")
    print(f"   ✅ No config objects needed - use direct parameters")
    print(f"   ✅ Selective layer pruning with names and regex patterns")
    print(f"   ✅ Detailed reporting of which layers were affected")
    print(f"   ✅ Flexible analysis and comparison tools")

if __name__ == "__main__":
    main()
