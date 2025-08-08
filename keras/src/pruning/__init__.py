"""Model pruning API for Keras."""

from keras.src.pruning.config import PruningConfig
from keras.src.pruning.core import apply_pruning_to_layer
from keras.src.pruning.core import apply_pruning_to_model
from keras.src.pruning.core import get_model_sparsity
from keras.src.pruning.core import match_layers_by_patterns
from keras.src.pruning.core import should_prune_layer
from keras.src.pruning.pruning_method import L1Pruning
from keras.src.pruning.pruning_method import LnPruning
from keras.src.pruning.pruning_method import PruningMethod
from keras.src.pruning.pruning_method import RandomPruning
from keras.src.pruning.pruning_method import SaliencyPruning
from keras.src.pruning.pruning_method import StructuredPruning
from keras.src.pruning.pruning_method import TaylorPruning
from keras.src.pruning.pruning_schedule import ConstantSparsity
from keras.src.pruning.pruning_schedule import LinearDecay
from keras.src.pruning.pruning_schedule import PolynomialDecay
from keras.src.pruning.pruning_schedule import PruningSchedule
from keras.src.pruning.pruning_utils import analyze_sparsity
from keras.src.pruning.pruning_utils import benchmark_inference
from keras.src.pruning.pruning_utils import compare_inference_speed
from keras.src.pruning.pruning_utils import compare_sparsity
from keras.src.pruning.pruning_utils import complete_pruning_analysis
from keras.src.pruning.pruning_utils import print_benchmark_report
from keras.src.pruning.pruning_utils import print_sparsity_report

# Public API
__all__ = [
    "PruningConfig",
    "get_model_sparsity",
    "should_prune_layer",
    "apply_pruning_to_model",
    "apply_pruning_to_layer",
    "match_layers_by_patterns",
    "PruningMethod",
    "StructuredPruning",
    "RandomPruning",
    "L1Pruning",
    "LnPruning",
    "SaliencyPruning",
    "TaylorPruning",
    "PruningSchedule",
    "ConstantSparsity",
    "PolynomialDecay",
    "LinearDecay",
    # Pruning analysis utilities
    "analyze_sparsity",
    "compare_sparsity",
    "print_sparsity_report",
    "benchmark_inference",
    "compare_inference_speed",
    "print_benchmark_report",
    "complete_pruning_analysis",
]
