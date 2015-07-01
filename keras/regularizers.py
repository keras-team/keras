from __future__ import absolute_import
import theano.tensor as T

class Regularizer(object):
    def set_param(self, p):
        self.p = p

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        return loss


class WeightRegularizer(Regularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        loss += T.sum(abs(self.p)) * self.l1
        loss += T.sum(self.p ** 2) * self.l2
        return loss


class ActivityRegularizer(Regularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        loss += self.l1 * T.sum(T.mean(abs(self.layer.get_output(True)), axis=0))
        loss += self.l2 * T.sum(T.mean(self.layer.get_output(True) ** 2, axis=0))
        return loss


def l1(l=0.01):
    return WeightRegularizer(l1=l)

def l2(l=0.01):
    return WeightRegularizer(l2=l)

def l1l2(l1=0.01, l2=0.01):
    return WeightRegularizer(l1=l1, l2=l2)

def activity_l1(l=0.01):
    return ActivityRegularizer(l1=l)

def activity_l2(l=0.01):
    return ActivityRegularizer(l2=l)

def activity_l1l2(l1=0.01, l2=0.01):
    return ActivityRegularizer(l1=l1, l2=l2)

identity = Regularizer
