"""Pruning callbacks for gradual weight pruning during training."""

from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.pruning.config import PruningConfig
from keras.src.pruning.core import apply_pruning_to_layer
from keras.src.pruning.core import get_model_sparsity


@keras_export("keras.callbacks.PruningCallback")
class PruningCallback(Callback):
    """Callback to gradually prune model weights during training.

    Args:
        config: PruningConfig instance with pruning parameters.
        verbose: Boolean. Whether to print progress messages.

    Example:
        ```python
        config = keras.pruning.PruningConfig(
            sparsity=0.8,
            method="magnitude",
            schedule="polynomial",
            start_step=100,
            end_step=1000,
            frequency=50
        )
        callback = keras.callbacks.PruningCallback(config, verbose=True)
        model.fit(x, y, callbacks=[callback])
        ```
    """

    def __init__(self, config, verbose=True):
        super().__init__()
        if not isinstance(config, PruningConfig):
            raise ValueError("config must be a PruningConfig instance")

        self.config = config
        self.verbose = verbose
        self.current_step = 0

    def on_train_batch_end(self, batch, logs=None):
        """Apply pruning at specified intervals."""
        self.current_step += 1

        if self.config.should_prune_at_step(self.current_step):
            current_sparsity = self.config.get_sparsity_for_step(
                self.current_step
            )

            # Apply pruning to all eligible layers
            pruned_layers = 0
            for layer in self.model.layers:
                if apply_pruning_to_layer(
                    layer, current_sparsity, self.config.method
                ):
                    pruned_layers += 1

            if self.verbose and pruned_layers > 0:
                actual_sparsity = get_model_sparsity(self.model)
                print(
                    f"Step {self.current_step}: Pruned {pruned_layers} layers "
                    f"(target: {current_sparsity:.3f}, actual: {actual_sparsity:.3f})"
                )

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
        config: PruningConfig instance with pruning parameters.
        verbose: Boolean. Whether to print progress messages.

    Example:
        ```python
        config = keras.pruning.PruningConfig(sparsity=0.6, method="structured")
        callback = keras.callbacks.PostTrainingPruning(config, verbose=True)
        model.fit(x, y, callbacks=[callback])
        ```
    """

    def __init__(self, config, verbose=True):
        super().__init__()
        if not isinstance(config, PruningConfig):
            raise ValueError("config must be a PruningConfig instance")

        self.config = config
        self.verbose = verbose

    def on_train_end(self, logs=None):
        """Apply pruning at the end of training."""
        if self.verbose:
            initial_sparsity = get_model_sparsity(self.model)
            print("Applying post-training pruning...")

        # Apply pruning to all eligible layers
        pruned_layers = 0
        for layer in self.model.layers:
            if apply_pruning_to_layer(
                layer, self.config.sparsity, self.config.method
            ):
                pruned_layers += 1

        if self.verbose:
            final_sparsity = get_model_sparsity(self.model)
            print(
                f"Post-training pruning complete. Pruned {pruned_layers} layers. "
                f"Sparsity: {initial_sparsity:.3f} -> {final_sparsity:.3f}"
            )
