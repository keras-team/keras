"""Model pruning API for Keras."""

from keras.src.pruning.config import PruningConfig
from keras.src.pruning.core import (
    get_model_sparsity,
    should_prune_layer,
    apply_pruning_to_model,
    apply_pruning_to_layer,
)
from keras.src.pruning.pruning_method import (
    PruningMethod,
    MagnitudePruning,
    StructuredPruning,
    RandomPruning,
)
from keras.src.pruning.pruning_schedule import (
    PruningSchedule,
    ConstantSparsity,
    PolynomialDecay,
    LinearDecay,
)

# Public API
__all__ = [
    "PruningConfig",
    "get_model_sparsity", 
    "should_prune_layer",
    "apply_pruning_to_model",
    "apply_pruning_to_layer",
    "PruningMethod",
    "MagnitudePruning",
    "StructuredPruning",
    "RandomPruning",
    "PruningSchedule",
    "ConstantSparsity", 
    "PolynomialDecay",
    "LinearDecay",
]
