"""Configuration classes for pruning."""

from keras.src.api_export import keras_export


@keras_export("keras.pruning.PruningConfig")
class PruningConfig:
    """Configuration for model pruning.

    Args:
        sparsity: Float between 0 and 1. Target sparsity level.
        method: String. Pruning method ('magnitude' or 'structured').
        schedule: PruningSchedule instance, or string for built-in schedules.
        start_step: Integer. Step to start pruning (used if schedule is string).
        end_step: Integer. Step to end pruning (used if schedule is string).
        frequency: Integer. How often to apply pruning (used if schedule is string).
        power: Float. Power for polynomial schedule (used if schedule is string).
        dataset: Optional data for advanced pruning methods (saliency, taylor).
        loss_fn: Optional loss function for gradient-based pruning methods.
        n: Float. Order for Ln norm (used with ln method).

    Example:
        ```python
        # Using built-in schedule
        config = PruningConfig(sparsity=0.8, schedule="polynomial", start_step=100, end_step=1000)

        # Using custom schedule
        custom_schedule = PolynomialDecay(start_step=50, end_step=500, power=2)
        config = PruningConfig(sparsity=0.8, schedule=custom_schedule)

        # Using saliency pruning with dataset
        config = PruningConfig(sparsity=0.5, method="saliency", dataset=train_dataset)
        ```
    """

    def __init__(
        self,
        sparsity=0.5,
        method="l1",
        schedule="constant",
        start_step=0,
        end_step=1000,
        frequency=100,
        power=3.0,
        dataset=None,
        loss_fn=None,
        n=2.0,
    ):
        if not 0 <= sparsity <= 1:
            raise ValueError(
                f"sparsity must be between 0 and 1. Got: {sparsity}"
            )

        # Accept string method names or PruningMethod instances
        valid_methods = [
            "magnitude",
            "l1",
            "structured",
            "l1_structured",
            "l2",
            "l2_structured",
            "saliency",
            "taylor",
        ]

        if isinstance(method, str) and method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods} or a PruningMethod instance. Got: {method}"
            )

        self.sparsity = sparsity
        self.method = method
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.n = n

        # Handle schedule - can be string or PruningSchedule instance
        if isinstance(schedule, str):
            from keras.src.pruning.pruning_schedule import ConstantSparsity
            from keras.src.pruning.pruning_schedule import PolynomialDecay

            if schedule == "constant":
                self.schedule = ConstantSparsity(
                    sparsity=sparsity,
                    start_step=start_step,
                    end_step=end_step,
                    frequency=frequency,
                )
            elif schedule == "polynomial":
                self.schedule = PolynomialDecay(
                    initial_sparsity=0.0,
                    target_sparsity=sparsity,
                    power=power,
                    start_step=start_step,
                    end_step=end_step,
                    frequency=frequency,
                )
            else:
                raise ValueError(
                    f"schedule must be 'constant', 'polynomial', or a PruningSchedule instance. Got: {schedule}"
                )
        else:
            # Assume it's a PruningSchedule instance
            self.schedule = schedule

    def get_sparsity_for_step(self, step):
        """Get the target sparsity for a given step."""
        return self.schedule.get_sparsity(step)

    def should_prune_at_step(self, step):
        """Determine if pruning should be applied at the given step."""
        return self.schedule.should_prune(step)
