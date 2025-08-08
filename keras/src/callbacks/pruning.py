"""Pruning callbacks for gradual weight pruning during training."""

import warnings

from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.pruning.core import apply_pruning_to_layer
from keras.src.pruning.core import apply_pruning_to_model
from keras.src.pruning.core import get_model_sparsity


@keras_export("keras.callbacks.PruningCallback")
class PruningCallback(Callback):
    """Callback to gradually prune model weights during training.

    Args:
        sparsity: Target sparsity (0-1) to reach by end_step.
        method: Pruning method - string name or PruningMethod instance.
        start_step: Step to start pruning (default: 100).
        end_step: Step to finish reaching target sparsity (default: 1000).
        frequency: How often to apply pruning (default: 50 steps).
        schedule: Sparsity schedule - "constant" or "polynomial" (default: "polynomial").
        layers_to_prune: Optional specification of which layers to prune.
        dataset: Dataset for gradient-based methods (tuple of (x, y)).
        loss_fn: Loss function for gradient-based methods.
        verbose: Boolean. Whether to print progress messages.

    Examples:
        ```python
        # Basic magnitude pruning
        callback = keras.callbacks.PruningCallback(
            sparsity=0.8,
            method="l1",
            start_step=100,
            end_step=1000,
            frequency=50,
            verbose=True
        )
        
        # Structured pruning on specific layers
        callback = keras.callbacks.PruningCallback(
            sparsity=0.6,
            method="structured",
            layers_to_prune=["conv.*", "dense_[0-9]"],  # Regex patterns
            start_step=200,
            end_step=800,
            verbose=True
        )
        
        # Saliency-based pruning with dataset
        callback = keras.callbacks.PruningCallback(
            sparsity=0.7,
            method="saliency",
            dataset=(x_train_sample, y_train_sample),
            loss_fn="categorical_crossentropy",
            frequency=100,
            verbose=True
        )
        
        model.fit(x, y, callbacks=[callback])
        ```
    """

    def __init__(self, sparsity=0.5, method="l1", start_step=100, end_step=1000,
                 frequency=50, schedule="polynomial", layers_to_prune=None,
                 dataset=None, loss_fn=None, verbose=True, **kwargs):
        super().__init__()
        
        # Use direct parameters
        self.sparsity = sparsity
        self.method = method
        self.start_step = start_step
        self.end_step = end_step
        self.frequency = frequency
        self.schedule = schedule
        self.layers_to_prune = layers_to_prune
        self.dataset = dataset
        self.loss_fn = loss_fn
        
        self.verbose = verbose
        self.current_step = 0
        self.kwargs = kwargs

    def should_prune_at_step(self, step):
        """Determine if pruning should be applied at this step."""
        if step < self.start_step:
            return False
        if step > self.end_step:
            return False
        return (step - self.start_step) % self.frequency == 0

    def get_sparsity_for_step(self, step):
        """Calculate target sparsity for the current step."""
        if step <= self.start_step:
            return 0.0
        if step >= self.end_step:
            return self.sparsity
        
        if self.schedule == "constant":
            return self.sparsity
        elif self.schedule == "polynomial":
            progress = (step - self.start_step) / (self.end_step - self.start_step)
            # Polynomial decay: gradually increase sparsity
            return self.sparsity * (progress ** 3)
        else:
            return self.sparsity

    def on_train_batch_end(self, batch, logs=None):
        """Apply pruning at specified intervals."""
        self.current_step += 1

        if self.should_prune_at_step(self.current_step):
            current_sparsity = self.get_sparsity_for_step(self.current_step)

            # Apply pruning to specified layers
            stats = apply_pruning_to_model(
                model=self.model,
                sparsity=current_sparsity,
                method=self.method,
                layers_to_prune=self.layers_to_prune,
                dataset=self.dataset,
                loss_fn=self.loss_fn,
                **self.kwargs
            )

            if self.verbose and stats["pruned_layers"] > 0:
                actual_sparsity = stats["final_sparsity"]
                print(
                    f"Step {self.current_step}: Pruned {stats['pruned_layers']} layers "
                    f"(target: {current_sparsity:.3f}, actual: {actual_sparsity:.3f})"
                )
                
                # Show which layers were pruned if layer selection was used
                if self.layers_to_prune is not None and "layers_pruned" in stats:
                    print(f"  Layers pruned: {', '.join(stats['layers_pruned'])}")

    def on_train_end(self, logs=None):
        """Print final sparsity when training ends."""
        if self.verbose:
            final_sparsity = get_model_sparsity(self.model)
            print(
                f"Training complete. Final model sparsity: {final_sparsity:.3f}"
            )


@keras_export("keras.callbacks.PostTrainingPruning")
class PostTrainingPruning(Callback):
    """Callback to apply pruning once at the end of training.

    Args:
        sparsity: Target sparsity (0-1) to apply.
        method: Pruning method - string name or PruningMethod instance.
        layers_to_prune: Optional specification of which layers to prune.
        dataset: Dataset for gradient-based methods (tuple of (x, y)).
        loss_fn: Loss function for gradient-based methods.
        verbose: Boolean. Whether to print progress messages.

    Examples:
        ```python
        # Basic structured pruning
        callback = keras.callbacks.PostTrainingPruning(
            sparsity=0.6, 
            method="structured", 
            verbose=True
        )
        
        # Selective layer pruning
        callback = keras.callbacks.PostTrainingPruning(
            sparsity=0.4,
            method="l1",
            layers_to_prune=["dense_1", "conv2d_.*"],  # Mix of names and patterns
            verbose=True
        )
        
        model.fit(x, y, callbacks=[callback])
        ```
    """

    def __init__(self, sparsity=0.5, method="l1", layers_to_prune=None,
                 dataset=None, loss_fn=None, verbose=True, **kwargs):
        super().__init__()
        
        # Use direct parameters
        self.sparsity = sparsity
        self.method = method
        self.layers_to_prune = layers_to_prune
        self.dataset = dataset
        self.loss_fn = loss_fn
        
        self.verbose = verbose
        self.kwargs = kwargs

    def on_train_end(self, logs=None):
        """Apply pruning at the end of training."""
        if self.verbose:
            initial_sparsity = get_model_sparsity(self.model)
            print("Applying post-training pruning...")

        # Apply pruning to specified layers
        stats = apply_pruning_to_model(
            model=self.model,
            sparsity=self.sparsity,
            method=self.method,
            layers_to_prune=self.layers_to_prune,
            dataset=self.dataset,
            loss_fn=self.loss_fn,
            **self.kwargs
        )

        if self.verbose:
            final_sparsity = stats["final_sparsity"]
            print(
                f"Post-training pruning complete. Pruned {stats['pruned_layers']} layers. "
                f"Sparsity: {initial_sparsity:.3f} -> {final_sparsity:.3f}"
            )
            
            # Show which layers were pruned if layer selection was used
            if self.layers_to_prune is not None and "layers_pruned" in stats:
                print(f"Layers pruned: {', '.join(stats['layers_pruned'])}")
                if "layers_skipped" in stats and stats["layers_skipped"]:
                    print(f"Layers skipped: {', '.join(stats['layers_skipped'])}")
