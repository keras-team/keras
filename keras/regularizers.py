from __future__ import absolute_import
import theano.tensor as T


class Regularizer(object):

    def update_gradient(self, gradient, params):
        raise NotImplementedError

    def update_loss(self, loss):
        raise NotImplementedError


class WeightsL1(Regularizer):
    def __init__(self, l=0.01):
        self.l = l

    def update_gradient(self, gradient, params):
        gradient += T.sgn(params) * self.l
        return gradient


class WeightsL2(Regularizer):
    def __init__(self, l=0.01):
        self.l = l

    def update_gradient(self, gradient, params):
        gradient += params * self.l
        return gradient


class WeightsL1L2(Regularizer):
    def __init__(self, l1=0.01, l2=0.01):
        self.l1 = l1
        self.l2 = l2

    def update_gradient(self, gradient, params):
        gradient += params * self.l2
        gradient += T.sgn(params) * self.l1
        return gradient


class Identity(Regularizer):

    def update_gradient(self, gradient, params):
        return gradient


class ActivityL1(Regularizer):
    def __init__(self, l = 0.01):
        self.l = l
        self.layer = None

    def set_layer(self, layer):
        self.layer = layer

    def update_loss(self, loss):
        return loss + self.l * T.sum(T.mean(abs(self.layer.get_output(True)), axis=0))


class ActivityL2(Regularizer):
    def __init__(self, l = 0.01):
        self.l = l
        self.layer = None

    def set_layer(self, layer):
        self.layer = layer

    def update_loss(self, loss):
        return loss + self.l * T.sum(T.mean(self.layer.get_output(True) ** 2, axis=0))


#old style variables for backwards compatibility
l1 = weights_l1 = WeightsL1
l2 = weights_l2 = WeightsL2
l1l2 = weights_l1l2 = WeightsL1L2
identity = Identity()

activity_l1 = ActivityL1
activity_l2 = ActivityL2
