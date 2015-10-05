from __future__ import absolute_import
from __future__ import print_function
import numpy as np


class LearningRateSchedule(object):
    def __init__(self, lr=0.01, start=0, **kwargs):
        self.__dict__.update(kwargs)
        self.lr = lr
        self.start = start

    def get_learning_rate(self, current_lr, iterations):
        pass

    def get_config(self, verbose=0):
        return {"name": self.__class__.__name__,
                "lr": self.lr,
                "start": self.start
                }


class FixedLearningRate(LearningRateSchedule):
    def __init__(self, lr=0.01, start=0, *args, **kwargs):
        super(FixedLearningRate, self).__init__(lr=lr, start=start, **kwargs)
        self.__dict__.update(locals())

    def get_learning_rate(self, current_lr, iterations):
        if iterations >= self.start:
            return self.lr
        else:
            return current_lr

    def get_config(self, verbose=0):
        return {"name": self.__class__.__name__,
                "lr": self.lr,
                "start": self.start
                }


class StepLearningRate(LearningRateSchedule):
    def __init__(self, lr=0.01, start=0, step_size=1, gamma=1., *args, **kwargs):
        super(StepLearningRate, self).__init__(lr=lr, start=start, **kwargs)
        self.__dict__.update(locals())

    def get_learning_rate(self, current_lr, iterations):
        if iterations >= self.start and self.step_size != 0:
            return self.lr * np.power(self.gamma, np.floor(iterations / self.step_size))

        return current_lr

    def get_config(self, verbose=0):
        return {"name": self.__class__.__name__,
                "lr": self.lr,
                "start": self.start,
                "step_size": self.step_size,
                "gamma": self.gamma
                }


class StagedLearningRate(LearningRateSchedule):
    def __init__(self, lr=0.01, start=0, stages=None, *args, **kwargs):
        super(StagedLearningRate, self).__init__(lr=lr, start=start, **kwargs)
        self.__dict__.update(locals())

    def get_learning_rate(self, current_lr, iterations):
        if iterations >= self.start and self.stages:
            if isinstance(self.stages, list) and len(self.stages) > iterations:
                return self.stages[iterations]
            elif hasattr(self.stages, '__call__'):
                return self.stages(iterations)

        return current_lr

    def get_config(self, verbose=0):
        return {"name": self.__class__.__name__,
                "lr": self.lr,
                "start": self.start,
                "stages": self.stages
                }


class ExpLearningRate(LearningRateSchedule):
    def __init__(self, lr=0.01, start=0, gamma=0., *args, **kwargs):
        super(ExpLearningRate, self).__init__(lr=lr, start=start, **kwargs)
        self.__dict__.update(locals())

    def get_learning_rate(self, current_lr, iterations):
        if iterations >= self.start:
            return self.lr * np.power(self.gamma, iterations)

        return current_lr

    def get_config(self, verbose=0):
        return {"name": self.__class__.__name__,
                "lr": self.lr,
                "start": self.start,
                "gamma": self.gamma
                }


class InvLearningRate(LearningRateSchedule):
    def __init__(self, lr=0.01, start=0, gamma=0., power=0., *args, **kwargs):
        super(InvLearningRate, self).__init__(lr=lr, start=start, **kwargs)
        self.__dict__.update(locals())

    def get_learning_rate(self, current_lr, iterations):
        if iterations >= self.start:
            return self.lr * np.power(1 + self.gamma * iterations, -self.power)

        return current_lr

    def get_config(self, verbose=0):
        return {"name": self.__class__.__name__,
                "lr": self.lr,
                "start": self.start,
                "gamma": self.gamma,
                "power": self.power
                }


class TriangularLearningRate(LearningRateSchedule):
    """
        Cyclical Learning Rates, Paper: http://arxiv.org/pdf/1506.01186.pdf
    """
    def __init__(self, lr=0.01, start=0, step_size=0, max_lr=0., *args, **kwargs):
        super(TriangularLearningRate, self).__init__(lr=lr, start=start, **kwargs)
        self.__dict__.update(locals())

    def get_learning_rate(self, current_lr, iterations):
        if iterations >= self.start:
            itr = iterations - self.start
            cycle = 1 + itr / (2 * self.step_size)
            if itr > 0:
                x = float(itr - (2 * cycle - 1) * self.step_size)
                x = x / self.step_size
                return self.lr + (self.max_lr - self.lr) * max(0.0, (1.0 - abs(x))/cycle)

        return current_lr

    def get_config(self, verbose=0):
        return {"name": self.__class__.__name__,
                "lr": self.lr,
                "start": self.start,
                "step_size": self.step_size,
                "max_lr": self.max_lr
                }
