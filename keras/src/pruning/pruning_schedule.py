"""Pruning schedule classes for controlling sparsity over time."""

from abc import ABC
from abc import abstractmethod

from keras.src.api_export import keras_export


@keras_export("keras.pruning.PruningSchedule")
class PruningSchedule(ABC):
    """Abstract base class for pruning schedules.

    A pruning schedule determines when pruning should occur and what sparsity
    level should be targeted at each training step.

    Args:
        start_step: Integer. Step to start pruning.
        end_step: Integer. Step to end pruning.
        frequency: Integer. How often to apply pruning in steps.
    """

    def __init__(self, start_step=0, end_step=1000, frequency=100):
        self.start_step = start_step
        self.end_step = end_step
        self.frequency = frequency

    def should_prune(self, step):
        """Determine if pruning should be applied at the given step.

        Args:
            step: Current training step.

        Returns:
            Boolean indicating whether to prune at this step.
        """
        if step < self.start_step or step > self.end_step:
            return False
        return (step - self.start_step) % self.frequency == 0

    @abstractmethod
    def get_sparsity(self, step):
        """Get the target sparsity for a given step.

        Args:
            step: Current training step.

        Returns:
            Float between 0 and 1 representing target sparsity.
        """
        pass


@keras_export("keras.pruning.ConstantSparsity")
class ConstantSparsity(PruningSchedule):
    """Constant sparsity schedule.

    Maintains the same sparsity level throughout the pruning period.

    Args:
        sparsity: Float between 0 and 1. Target sparsity level.
        start_step: Integer. Step to start pruning.
        end_step: Integer. Step to end pruning.
        frequency: Integer. How often to apply pruning in steps.
    """

    def __init__(self, sparsity, start_step=0, end_step=1000, frequency=100):
        super().__init__(start_step, end_step, frequency)
        if not 0 <= sparsity <= 1:
            raise ValueError(
                f"sparsity must be between 0 and 1. Got: {sparsity}"
            )
        self.sparsity = sparsity

    def get_sparsity(self, step):
        """Returns constant sparsity level."""
        if self.start_step <= step <= self.end_step:
            return self.sparsity
        return 0.0


@keras_export("keras.pruning.PolynomialDecay")
class PolynomialDecay(PruningSchedule):
    """Polynomial decay sparsity schedule.

    Gradually increases sparsity from initial to target using polynomial decay.

    Args:
        initial_sparsity: Float between 0 and 1. Initial sparsity level.
        target_sparsity: Float between 0 and 1. Target sparsity level.
        power: Float. Power for polynomial decay (higher = more aggressive).
        start_step: Integer. Step to start pruning.
        end_step: Integer. Step to end pruning.
        frequency: Integer. How often to apply pruning in steps.
    """

    def __init__(
        self,
        initial_sparsity=0.0,
        target_sparsity=0.8,
        power=3.0,
        start_step=0,
        end_step=1000,
        frequency=100,
    ):
        super().__init__(start_step, end_step, frequency)

        if not 0 <= initial_sparsity <= 1:
            raise ValueError(
                f"initial_sparsity must be between 0 and 1. Got: {initial_sparsity}"
            )
        if not 0 <= target_sparsity <= 1:
            raise ValueError(
                f"target_sparsity must be between 0 and 1. Got: {target_sparsity}"
            )
        if initial_sparsity >= target_sparsity:
            raise ValueError(
                f"initial_sparsity must be less than target_sparsity. "
                f"Got: {initial_sparsity} >= {target_sparsity}"
            )

        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.power = power

    def get_sparsity(self, step):
        """Returns sparsity level based on polynomial decay."""
        if step < self.start_step:
            return self.initial_sparsity
        if step >= self.end_step:
            return self.target_sparsity

        # Calculate progress as a value between 0 and 1
        progress = (step - self.start_step) / (self.end_step - self.start_step)

        # Apply polynomial decay
        sparsity_range = self.target_sparsity - self.initial_sparsity
        current_sparsity = self.initial_sparsity + sparsity_range * (
            progress**self.power
        )

        return current_sparsity


@keras_export("keras.pruning.LinearDecay")
class LinearDecay(PruningSchedule):
    """Linear decay sparsity schedule.

    Gradually increases sparsity from initial to target linearly.

    Args:
        initial_sparsity: Float between 0 and 1. Initial sparsity level.
        target_sparsity: Float between 0 and 1. Target sparsity level.
        start_step: Integer. Step to start pruning.
        end_step: Integer. Step to end pruning.
        frequency: Integer. How often to apply pruning in steps.
    """

    def __init__(
        self,
        initial_sparsity=0.0,
        target_sparsity=0.8,
        start_step=0,
        end_step=1000,
        frequency=100,
    ):
        super().__init__(start_step, end_step, frequency)

        if not 0 <= initial_sparsity <= 1:
            raise ValueError(
                f"initial_sparsity must be between 0 and 1. Got: {initial_sparsity}"
            )
        if not 0 <= target_sparsity <= 1:
            raise ValueError(
                f"target_sparsity must be between 0 and 1. Got: {target_sparsity}"
            )
        if initial_sparsity >= target_sparsity:
            raise ValueError(
                f"initial_sparsity must be less than target_sparsity. "
                f"Got: {initial_sparsity} >= {target_sparsity}"
            )

        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity

    def get_sparsity(self, step):
        """Returns sparsity level based on linear interpolation."""
        if step < self.start_step:
            return self.initial_sparsity
        if step >= self.end_step:
            return self.target_sparsity

        # Linear interpolation
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        sparsity_range = self.target_sparsity - self.initial_sparsity
        current_sparsity = self.initial_sparsity + sparsity_range * progress

        return current_sparsity


@keras_export("keras.pruning.ConstantSparsity")
class ConstantSparsity(PruningSchedule):
    """Constant sparsity schedule.

    Maintains the same sparsity level throughout the pruning period.

    Args:
        sparsity: Float between 0 and 1. Target sparsity level.
        start_step: Integer. Step to start pruning.
        end_step: Integer. Step to end pruning.
        frequency: Integer. How often to apply pruning in steps.
    """

    def __init__(self, sparsity, start_step=0, end_step=1000, frequency=100):
        super().__init__(start_step, end_step, frequency)
        if not 0 <= sparsity <= 1:
            raise ValueError(
                f"sparsity must be between 0 and 1. Got: {sparsity}"
            )
        self.sparsity = sparsity

    def get_sparsity(self, step):
        """Returns constant sparsity level."""
        if self.start_step <= step <= self.end_step:
            return self.sparsity
        return 0.0


@keras_export("keras.pruning.PolynomialDecay")
class PolynomialDecay(PruningSchedule):
    """Polynomial decay sparsity schedule.

    Gradually increases sparsity from initial to target using polynomial decay.

    Args:
        initial_sparsity: Float between 0 and 1. Initial sparsity level.
        target_sparsity: Float between 0 and 1. Target sparsity level.
        power: Float. Power for polynomial decay (higher = more aggressive).
        start_step: Integer. Step to start pruning.
        end_step: Integer. Step to end pruning.
        frequency: Integer. How often to apply pruning in steps.
    """

    def __init__(
        self,
        initial_sparsity=0.0,
        target_sparsity=0.8,
        power=3.0,
        start_step=0,
        end_step=1000,
        frequency=100,
    ):
        super().__init__(start_step, end_step, frequency)

        if not 0 <= initial_sparsity <= 1:
            raise ValueError(
                f"initial_sparsity must be between 0 and 1. Got: {initial_sparsity}"
            )
        if not 0 <= target_sparsity <= 1:
            raise ValueError(
                f"target_sparsity must be between 0 and 1. Got: {target_sparsity}"
            )
        if initial_sparsity >= target_sparsity:
            raise ValueError(
                f"initial_sparsity must be less than target_sparsity. "
                f"Got: {initial_sparsity} >= {target_sparsity}"
            )

        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.power = power

    def get_sparsity(self, step):
        """Returns sparsity level based on polynomial decay."""
        if step < self.start_step:
            return self.initial_sparsity
        if step >= self.end_step:
            return self.target_sparsity

        # Calculate progress as a value between 0 and 1
        progress = (step - self.start_step) / (self.end_step - self.start_step)

        # Apply polynomial decay
        sparsity_range = self.target_sparsity - self.initial_sparsity
        current_sparsity = self.initial_sparsity + sparsity_range * (
            progress**self.power
        )

        return current_sparsity


@keras_export("keras.pruning.LinearDecay")
class LinearDecay(PruningSchedule):
    """Linear decay sparsity schedule.

    Gradually increases sparsity from initial to target linearly.

    Args:
        initial_sparsity: Float between 0 and 1. Initial sparsity level.
        target_sparsity: Float between 0 and 1. Target sparsity level.
        start_step: Integer. Step to start pruning.
        end_step: Integer. Step to end pruning.
        frequency: Integer. How often to apply pruning in steps.
    """

    def __init__(
        self,
        initial_sparsity=0.0,
        target_sparsity=0.8,
        start_step=0,
        end_step=1000,
        frequency=100,
    ):
        super().__init__(start_step, end_step, frequency)

        if not 0 <= initial_sparsity <= 1:
            raise ValueError(
                f"initial_sparsity must be between 0 and 1. Got: {initial_sparsity}"
            )
        if not 0 <= target_sparsity <= 1:
            raise ValueError(
                f"target_sparsity must be between 0 and 1. Got: {target_sparsity}"
            )
        if initial_sparsity >= target_sparsity:
            raise ValueError(
                f"initial_sparsity must be less than target_sparsity. "
                f"Got: {initial_sparsity} >= {target_sparsity}"
            )

        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity

    def get_sparsity(self, step):
        """Returns sparsity level based on linear interpolation."""
        if step < self.start_step:
            return self.initial_sparsity
        if step >= self.end_step:
            return self.target_sparsity

        # Linear interpolation
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        sparsity_range = self.target_sparsity - self.initial_sparsity
        current_sparsity = self.initial_sparsity + sparsity_range * progress

        return current_sparsity
