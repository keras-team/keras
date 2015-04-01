import theano
import theano.tensor as T
import numpy as np

from utils.theano_utils import shared_zeros, shared_scalar

def clip_norm(g, c, n):
    if c > 0:
        g = T.switch(T.ge(n, c), g*c/n, g)
    return g

class Optimizer(object):
    def get_updates(self, params, grads):
        raise NotImplementedError

    def get_gradients(self, cost, params):
        grads = T.grad(cost, params)

        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = T.sqrt(sum([T.sum(g**2) for g in grads]))
            grads = [clip_norm(g, c, norm) for g in grads]

        new_grads = []
        for p, g in zip(params, grads):
            if hasattr(self, 'l1') and self.l1 > 0:
                g += T.sgn(p) * self.l1

            if hasattr(self, 'l2') and self.l2 > 0:
                g += p * self.l2

            if hasattr(self, 'maxnorm') and self.maxnorm > 0:
                norms = T.sqrt(T.sum(T.sqr(p), axis=0))
                desired = T.clip(norms, 0, self.maxnorm)
                p = p * (desired / (1e-7 + norms))

            new_grads.append(g)
        return new_grads


class SGD(Optimizer):

    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, *args, **kwargs):
        self.__dict__.update(locals())
        self.iterations = shared_scalar(0)

    def get_updates(self, params, cost):
        grads = self.get_gradients(cost, params)
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        updates = [(self.iterations, self.iterations+1.)]

        for p, g in zip(params, grads):
            m = shared_zeros(p.get_value().shape) # momentum
            v = self.momentum * m - lr * g # velocity
            updates.append((m, v)) 

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v
            updates.append((p, new_p))
        return updates


class RMSprop(Optimizer):

    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, *args, **kwargs):
        self.__dict__.update(locals())

    def get_updates(self, params, cost):
        grads = self.get_gradients(cost, params)
        accumulators = [shared_zeros(p.get_value().shape) for p in params]
        updates = []

        for p, g, a in zip(params, grads, accumulators):
            new_a = self.rho * a + (1 - self.rho) * g ** 2 # update accumulator
            updates.append((a, new_a))

            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            updates.append((p, new_p))
        return updates


class Adagrad(Optimizer):

    def __init__(self, lr=0.01, epsilon=1e-6, *args, **kwargs):
        self.__dict__.update(locals())

    def get_updates(self, params, cost):
        grads = self.get_gradients(cost, params)
        accumulators = [shared_zeros(p.get_value().shape) for p in params]
        updates = []

        for p, g, a in zip(params, grads, accumulators):
            new_a = a + g ** 2 # update accumulator
            updates.append((a, new_a))

            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            updates.append((p, new_p))
        return updates


class Adadelta(Optimizer):
    
    def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6, *args, **kwargs):
        self.__dict__.update(locals())

    def get_updates(self, params, cost):
        grads = self.get_gradients(cost, params)
        accumulators = [shared_zeros(p.get_value().shape) for p in params]
        delta_accumulators = [shared_zeros(p.get_value().shape) for p in params]
        updates = []

        for p, g, a, d_a in zip(params, grads, accumulators, delta_accumulators):
            new_a = self.rho * a + (1 - self.rho) * g ** 2 # update accumulator
            updates.append((a, new_a))

            # use the new accumulator and the *old* delta_accumulator
            update = g * T.sqrt(d_a + self.epsilon) / T.sqrt(new_a + self.epsilon)

            new_p = p - self.lr * update
            updates.append((p, new_p))

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * update ** 2
            updates.append((d_a, new_d_a))
        return updates

# aliases
sgd = SGD
rmsprop = RMSprop
adagrad = Adagrad
adadelta = Adadelta

from utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'optimizer', instantiate=True)
