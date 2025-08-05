"""Pruning schedule classes for controlling pruning timing."""


class PruningSchedule:
    """Base class for pruning schedules.
    
    A pruning schedule determines when pruning should be applied during training.
    """
    
    def __init__(
        self,
        start_step=0,
        end_step=None,
        frequency=100,
    ):
        """Initialize pruning schedule.
        
        Args:
            start_step: Int, step at which to start pruning.
            end_step: Int or None, step at which to end pruning.
                If None, continue pruning indefinitely.
            frequency: Int, how often to apply pruning in steps.
        """
        self.start_step = start_step
        self.end_step = end_step
        self.frequency = frequency
        
    def should_prune(self, step):
        """Determine if pruning should be applied at current step.
        
        Args:
            step: Current training step.
            
        Returns:
            Boolean indicating whether to prune.
        """
        if step < self.start_step:
            return False
            
        if self.end_step and step > self.end_step:
            return False
            
        return (step - self.start_step) % self.frequency == 0


class PolynomialDecay(PruningSchedule):
    """Pruning schedule with polynomial sparsity growth.
    
    Gradually increases sparsity from initial_sparsity to target_sparsity
    following a polynomial curve.
    """
    
    def __init__(
        self,
        initial_sparsity=0.0,
        target_sparsity=0.5,
        power=3,
        **kwargs
    ):
        """Initialize polynomial decay schedule.
        
        Args:
            initial_sparsity: Float, starting sparsity.
            target_sparsity: Float, final sparsity to reach.
            power: Int, power for polynomial curve.
            **kwargs: Arguments for base PruningSchedule.
        """
        super().__init__(**kwargs)
        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.power = power
        
    def get_sparsity(self, step):
        """Get target sparsity for current step.
        
        Args:
            step: Current training step.
            
        Returns:
            Target sparsity level.
        """
        if step < self.start_step:
            return self.initial_sparsity
            
        if self.end_step and step > self.end_step:
            return self.target_sparsity
            
        # Normalize step to [0, 1] range
        prog = (step - self.start_step) / (self.end_step - self.start_step)
        # Apply polynomial curve
        prog = prog ** self.power
        
        return self.initial_sparsity + (
            self.target_sparsity - self.initial_sparsity
        ) * prog
