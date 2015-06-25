from __future__ import absolute_import
import theano.tensor as T


class Regularizer(object):
    def __init__(self):
        self.modifies_gradient = False
        self.modifies_cost = False

    def update_gradient(self, gradient, params):
        raise NotImplementedError

    def update_cost(self, cost):
        raise NotImplementedError


class RegularizerWeightsL1(Regularizer):
    def __init__(self, l=0.01):
        super(Regularizer, self).__init__()

        self.l = l
        self.modifies_gradient = True

    def update_gradient(self, gradient, params):
        gradient += T.sgn(params) * self.l
        return gradient

    def __call__(self, gradient, params):
        return self.update_gradient(gradient, params)


class RegularizerWeightsL2(Regularizer):
    def __init__(self, l=0.01):
        super(Regularizer, self).__init__()

        self.l = l
        self.modifies_gradient = True

    def update_gradient(self, gradient, params):
        gradient += params * self.l
        return gradient

    def __call__(self, gradient, params):
        return self.update_gradient(gradient, params)


class RegularizerWeightsL1L2(Regularizer):
    def __init__(self, l1=0.01, l2=0.01):
        super(Regularizer, self).__init__()

        self.l1 = l1
        self.l2 = l2
        self.modifies_gradient = True

    def update_gradient(self, gradient, params):
        gradient += params * self.l2
        gradient += T.sgn(params) * self.l1
        return gradient

    def __call__(self, gradient, params):
        return self.update_gradient(gradient, params)

class RegularizerIdentity(Regularizer):
    def __init__(self):
        super(Regularizer, self).__init__()

        self.modifies_gradient = True

    def update_gradient(self, gradient, params):
        return gradient

    def __call__(self, gradient, params):
        return self.update_gradient(gradient, params)


class RegularizerActivityL1(Regularizer):
    def __init__(self, l = 0.01):
        super(Regularizer, self).__init__()

        self.l = l
        self.layer = None
        self.modifies_cost = True

    def set_layer(self, layer):
        self.layer = layer

    def update_cost(self, cost):
        return cost + self.l * T.sum(T.mean(abs(self.layer.get_output(True)), axis=0))

    def __call__(self, cost):
        return self.update_cost(cost)

class RegularizerActivityL2(Regularizer):
    def __init__(self, l = 0.01):
        super(Regularizer, self).__init__()

        self.l = l
        self.layer = None
        self.modifies_cost = True

    def set_layer(self, layer):
        self.layer = layer

    def update_cost(self, cost):
        return cost + self.l * T.sum(T.mean(self.layer.get_output(True) ** 2, axis=0))

    def __call__(self, cost):
        return self.update_cost(cost)

#old style variables for backwards compatibility
l1 = RegularizerWeightsL1
l2 = RegularizerWeightsL2
l1l2 = RegularizerWeightsL1L2
identity = RegularizerIdentity()

activity_l1 = RegularizerActivityL1
activity_l2 = RegularizerActivityL2
