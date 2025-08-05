"""Pruning callback for gradual weight pruning during training."""

from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.pruning import MagnitudePruning, StructuredPruning
from keras.src import backend
from keras.src import ops


def _get_model_sparsity(model):
    """Calculate the overall sparsity of a model."""
    total_weights = 0
    zero_weights = 0
    
    for layer in model.layers:
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            weights = layer.kernel.value
            total_weights += ops.size(weights)
            zero_weights += ops.sum(ops.cast(weights == 0, "int32"))
        
        if hasattr(layer, 'bias') and layer.bias is not None:
            bias = layer.bias.value
            total_weights += ops.size(bias)
            zero_weights += ops.sum(ops.cast(bias == 0, "int32"))
    
    if total_weights == 0:
        return 0.0
    return float(zero_weights / total_weights)


def _gradual_pruning_schedule(current_step, start_step, end_step, initial_sparsity=0.0, final_sparsity=0.9):
    """Compute sparsity ratio for gradual pruning schedule."""
    if current_step < start_step:
        return initial_sparsity
    if current_step >= end_step:
        return final_sparsity
    
    # Cubic sparsity schedule
    progress = (current_step - start_step) / (end_step - start_step)
    sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * (progress ** 3)
    
    return sparsity


@keras_export("keras.callbacks.PruningCallback")
class PruningCallback(Callback):
    """Callback to gradually prune model weights during training.
    
    This callback implements gradual magnitude-based or structured pruning
    following a polynomial decay schedule.
    
    Args:
        target_sparsity: Float between 0 and 1. Target sparsity level.
        start_step: Integer. Step to start pruning.
        end_step: Integer. Step to end pruning schedule.
        frequency: Integer. How often to apply pruning (in steps).
        method: String or PruningMethod instance. Pruning method to use.
        initial_sparsity: Float. Initial sparsity level.
        verbose: Boolean. Whether to print progress messages.
    """
    
    def __init__(
        self,
        target_sparsity=0.5,
        start_step=0,
        end_step=1000,
        frequency=100,
        method="magnitude",
        initial_sparsity=0.0,
        verbose=True
    ):
        super().__init__()
        self.target_sparsity = target_sparsity
        self.start_step = start_step
        self.end_step = end_step
        self.frequency = frequency
        self.initial_sparsity = initial_sparsity
        self.verbose = verbose
        self.current_step = 0
        
        # Initialize pruning method
        if isinstance(method, str):
            if method == "magnitude":
                self.pruning_method = MagnitudePruning()
            elif method == "structured":
                self.pruning_method = StructuredPruning()
            else:
                raise ValueError(f"Unknown pruning method: {method}")
        else:
            self.pruning_method = method
    
    def on_train_batch_end(self, batch, logs=None):
        """Apply pruning at specified intervals."""
        self.current_step += 1
        
        if (self.current_step >= self.start_step and 
            self.current_step <= self.end_step and
            self.current_step % self.frequency == 0):
            
            # Calculate current sparsity
            current_sparsity = _gradual_pruning_schedule(
                self.current_step,
                self.start_step,
                self.end_step,
                self.initial_sparsity,
                self.target_sparsity
            )
            
            # Apply pruning to all pruneable layers
            self._prune_model(current_sparsity)
            
            if self.verbose:
                actual_sparsity = _get_model_sparsity(self.model)
                print(
                    f"Step {self.current_step}: Applied pruning "
                    f"(target: {current_sparsity:.3f}, actual: {actual_sparsity:.3f})"
                )
    
    def _prune_model(self, sparsity_ratio):
        """Apply pruning to all eligible layers in the model."""
        for layer in self.model.layers:
            if self._should_prune_layer(layer):
                # Get current weights
                weights = layer.kernel.value
                
                # Compute and apply pruning mask
                mask = self.pruning_method.compute_mask(weights, sparsity_ratio)
                pruned_weights = self.pruning_method.apply_mask(weights, mask)
                
                # Update layer weights
                layer.kernel.assign(pruned_weights)
    
    def _should_prune_layer(self, layer):
        """Determine if a layer should be pruned."""
        # Only prune Dense and Conv layers with kernels
        layer_types = ('Dense', 'Conv1D', 'Conv2D', 'Conv3D', 'DepthwiseConv2D')
        return (
            layer.__class__.__name__ in layer_types and
            hasattr(layer, 'kernel') and 
            layer.kernel is not None
        )
    
    def on_train_end(self, logs=None):
        """Print final sparsity when training ends."""
        if self.verbose:
            final_sparsity = _get_model_sparsity(self.model)
            print(f"Training complete. Final model sparsity: {final_sparsity:.3f}")


@keras_export("keras.callbacks.PostTrainingPruning")  
class PostTrainingPruning(Callback):
    """Callback to apply pruning once at the end of training.
    
    Args:
        sparsity: Float between 0 and 1. Target sparsity level.
        method: String or PruningMethod instance. Pruning method to use.
        verbose: Boolean. Whether to print progress messages.
    """
    
    def __init__(self, sparsity=0.5, method="magnitude", verbose=True):
        super().__init__()
        self.sparsity = sparsity
        self.verbose = verbose
        
        # Initialize pruning method
        if isinstance(method, str):
            if method == "magnitude":
                self.pruning_method = MagnitudePruning()
            elif method == "structured":
                self.pruning_method = StructuredPruning()
            else:
                raise ValueError(f"Unknown pruning method: {method}")
        else:
            self.pruning_method = method
    
    def on_train_end(self, logs=None):
        """Apply pruning at the end of training."""
        if self.verbose:
            initial_sparsity = _get_model_sparsity(self.model)
            print(f"Applying post-training pruning...")
        
        # Apply pruning to all eligible layers
        for layer in self.model.layers:
            if self._should_prune_layer(layer):
                # Get current weights
                weights = layer.kernel.value
                
                # Compute and apply pruning mask
                mask = self.pruning_method.compute_mask(weights, self.sparsity)
                pruned_weights = self.pruning_method.apply_mask(weights, mask)
                
                # Update layer weights
                layer.kernel.assign(pruned_weights)
        
        if self.verbose:
            final_sparsity = _get_model_sparsity(self.model)
            print(
                f"Post-training pruning complete. "
                f"Sparsity: {initial_sparsity:.3f} -> {final_sparsity:.3f}"
            )
    
    def _should_prune_layer(self, layer):
        """Determine if a layer should be pruned."""
        layer_types = ('Dense', 'Conv1D', 'Conv2D', 'Conv3D', 'DepthwiseConv2D')
        return (
            layer.__class__.__name__ in layer_types and
            hasattr(layer, 'kernel') and 
            layer.kernel is not None
        )
