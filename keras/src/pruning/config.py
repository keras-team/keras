"""Configuration classes for pruning."""

from keras.src.api_export import keras_export


@keras_export("keras.pruning.PruningConfig")
class PruningConfig:
    """Configuration for model pruning.
    
    Args:
        sparsity: Float between 0 and 1. Target sparsity level.
        method: String. Pruning method ('magnitude' or 'structured').
        schedule: String. Pruning schedule ('constant' or 'polynomial').
        start_step: Integer. Step to start pruning (for schedules).
        end_step: Integer. Step to end pruning (for schedules).
        frequency: Integer. How often to apply pruning in steps.
        power: Float. Power for polynomial schedule.
    """
    
    def __init__(
        self,
        sparsity=0.5,
        method="magnitude",
        schedule="constant", 
        start_step=0,
        end_step=1000,
        frequency=100,
        power=3.0
    ):
        if not 0 <= sparsity <= 1:
            raise ValueError(f"sparsity must be between 0 and 1. Got: {sparsity}")
        
        if method not in ["magnitude", "structured"]:
            raise ValueError(f"method must be 'magnitude' or 'structured'. Got: {method}")
            
        if schedule not in ["constant", "polynomial"]:
            raise ValueError(f"schedule must be 'constant' or 'polynomial'. Got: {schedule}")
        
        self.sparsity = sparsity
        self.method = method
        self.schedule = schedule
        self.start_step = start_step
        self.end_step = end_step
        self.frequency = frequency
        self.power = power
    
    def get_sparsity_for_step(self, step):
        """Get the target sparsity for a given step."""
        if self.schedule == "constant":
            return self.sparsity
        elif self.schedule == "polynomial":
            if step < self.start_step:
                return 0.0
            if step >= self.end_step:
                return self.sparsity
            
            progress = (step - self.start_step) / (self.end_step - self.start_step)
            return self.sparsity * (progress ** self.power)
        
    def should_prune_at_step(self, step):
        """Determine if pruning should be applied at the given step."""
        if step < self.start_step or step > self.end_step:
            return False
        return (step - self.start_step) % self.frequency == 0
