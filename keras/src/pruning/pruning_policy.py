"""Base class for pruning policies."""

from keras.src.layers.layer import Layer


class PruningPolicy:
    """Base class for pruning policies.
    
    A pruning policy defines how and when to prune a model's weights.
    """

    def __init__(
        self,
        target_sparsity=0.5,
        method="magnitude",
        granularity="weight",
    ):
        """Initialize the pruning policy.
        
        Args:
            target_sparsity: Float between 0 and 1. Target sparsity level.
            method: String, pruning method to use. One of:
                - 'magnitude': Remove weights with smallest absolute value
                - 'random': Randomly remove weights
                - 'l1': Remove weights with smallest L1 norm
                - 'l2': Remove weights with smallest L2 norm
            granularity: String, level at which to apply pruning. One of:
                - 'weight': Prune individual weights
                - 'kernel': Prune entire kernels/filters
                - 'channel': Prune entire channels
        """
        self.target_sparsity = target_sparsity
        self.method = method
        self.granularity = granularity
        
    def get_mask(self, weights):
        """Generate pruning mask for weights.
        
        Args:
            weights: NumPy array of weights to prune.
            
        Returns:
            Binary mask of same shape as weights.
        """
        if self.method == "magnitude":
            return self._magnitude_mask(weights)
        elif self.method == "random":
            return self._random_mask(weights)
        elif self.method == "l1":
            return self._l1_mask(weights)
        elif self.method == "l2":
            return self._l2_mask(weights)
        else:
            raise ValueError(f"Unknown pruning method: {self.method}")
            
    def _magnitude_mask(self, weights):
        """Generate mask based on absolute weight values."""
        abs_weights = np.abs(weights)
        threshold = np.percentile(abs_weights, self.target_sparsity * 100)
        return abs_weights > threshold
        
    def _random_mask(self, weights):
        """Generate random pruning mask."""
        mask = np.random.uniform(0, 1, weights.shape) > self.target_sparsity
        return mask.astype(weights.dtype)
        
    def _l1_mask(self, weights):
        """Generate mask based on L1 norm."""
        if self.granularity == "kernel":
            norms = np.sum(np.abs(weights), axis=(0, 1))
        else:
            norms = np.abs(weights)
        threshold = np.percentile(norms, self.target_sparsity * 100)
        return norms > threshold
        
    def _l2_mask(self, weights):
        """Generate mask based on L2 norm."""
        if self.granularity == "kernel":
            norms = np.sqrt(np.sum(np.square(weights), axis=(0, 1)))
        else:
            norms = np.square(weights)
        threshold = np.percentile(norms, self.target_sparsity * 100)
        return norms > threshold
