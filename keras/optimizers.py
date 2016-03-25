from __future__ import absolute_import
from . import backend as K
import numpy as np
from .utils.generic_utils import get_from_module
from collections import defaultdict
from six.moves import zip


def clip_norm(g, c, n):
    if c > 0:
        g = K.switch(n >= c, g * c / n, g)
    return g


def kl_divergence(p, p_hat):
    return p_hat - p + p * K.log(p / p_hat)


def do_subtensor_update(grads):
    return len(grads) > 1 or list(grads)[0]


def tensor_set(tensor, grads, update_expr, *args):
    '''Wraps tensor and subtensor updates into one function'''
    if not do_subtensor_update(grads):
        g = grads.items()[0][1]
        tensor = update_expr(tensor, g, *args)
    else:
        for i, (indices, g) in enumerate(grads.items()):
            tensor_sub = tensor[indices]
            sliced_args = [arg[indices] for arg in args]

            update = update_expr(tensor_sub, g, *sliced_args)

            tensor = K.scatter_update(tensor, indices, update)

    return tensor


class Optimizer(object):
    '''Abstract optimizer base class.

    Note: this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.

    All Keras optimizers support the following keyword arguments:

        clipnorm: float >= 0. Gradients will be clipped
            when their L2 norm exceeds this value.
        clipvalue: float >= 0. Gradients will be clipped
            when their absolute value exceeds this value.
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.updates = []

    def get_state(self):
        return [K.get_value(u[0]) for u in self.updates]

    def set_state(self, value_list):
        assert len(self.updates) == len(value_list)
        for u, v in zip(self.updates, value_list):
            K.set_value(u[0], v)

    def get_updates(self, params, constraints, loss):
        raise NotImplementedError

    def get_gradients(self, loss, params):
        dict_to_list = []
        for param in params:
            for inner_param in params[param]:
                dict_to_list.append((param, inner_param, params[param][inner_param]))
        grads = K.gradients(loss, [item[2] for item in dict_to_list])
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        dict_of_grads = defaultdict(dict)
        for (key, inner_key, _), grad in zip(dict_to_list, grads):
            dict_of_grads[key][inner_key] = grad
        return dict_of_grads

    def get_config(self):
        return {"name": self.__class__.__name__}


class SGD(Optimizer):
    '''Stochastic gradient descent, with support for momentum,
    decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    '''
    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False,
                 *args, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0.)
        self.lr = K.variable(lr)
        self.momentum = K.variable(momentum)
        self.decay = K.variable(decay)

    def get_updates(self, params, constraints, loss):
        grad_dict = self.get_gradients(loss, params)
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        self.updates = [(self.iterations, self.iterations + 1.)]

        for p in grad_dict:
            m = K.variable(np.zeros(K.get_value(p).shape))  # momentum
            v = tensor_set(m, grad_dict[p],
                           lambda x, y: self.momentum * x - lr * y)  # velocity
            self.updates.append((m, v))

            if self.nesterov:
                new_p = tensor_set(p, grad_dict[p],
                                   lambda x, y, v: x + self.momentum * v - lr * y,
                                   v)
            else:
                new_p = tensor_set(p, grad_dict[p],
                                   lambda x, y, v: x + v,
                                   v)
            self.updates.append((p, constraints[p](new_p)))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(K.get_value(self.lr)),
                "momentum": float(K.get_value(self.momentum)),
                "decay": float(K.get_value(self.decay)),
                "nesterov": self.nesterov}


class RMSprop(Optimizer):
    '''RMSProp optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    This optimizer is usually a good choice for recurrent
    neural networks.

    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.
    '''
    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, *args, **kwargs):
        super(RMSprop, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = K.variable(lr)
        self.rho = K.variable(rho)

    def get_updates(self, params, constraints, loss):
        grad_dict = self.get_gradients(loss, params)
        param_list = list(params.keys())
        accumulators = [K.variable(np.zeros(K.get_value(p).shape)) for p in param_list]
        self.updates = []

        for p, a in zip(param_list, accumulators):
            # update accumulator
            new_a = tensor_set(a, grad_dict[p],
                               lambda x, y: self.rho * x + (1 - self.rho) * K.square(y))
            self.updates.append((a, new_a))

            new_p = tensor_set(p, grad_dict[p],
                               lambda x, y, a: x - self.lr * y / K.sqrt(a + self.epsilon),
                               new_a)
            self.updates.append((p, constraints[p](new_p)))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(K.get_value(self.lr)),
                "rho": float(K.get_value(self.rho)),
                "epsilon": self.epsilon}


class Adagrad(Optimizer):
    '''Adagrad optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Learning rate.
        epsilon: float >= 0.
    '''
    def __init__(self, lr=0.01, epsilon=1e-6, *args, **kwargs):
        super(Adagrad, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = K.variable(lr)

    def get_updates(self, params, constraints, loss):
        grad_dict = self.get_gradients(loss, params)
        param_list = list(params.keys())
        accumulators = [K.variable(np.zeros(K.get_value(p).shape)) for p in param_list]
        self.updates = []

        for p, a in zip(param_list, accumulators):
            new_a = tensor_set(a, grad_dict[p],
                               lambda x, y: x + K.square(y))
            self.updates.append((a, new_a))
            new_p = tensor_set(p, grad_dict[p],
                               lambda x, y, a: x - self.lr * y / K.sqrt(a + self.epsilon),
                               new_a)
            self.updates.append((p, constraints[p](new_p)))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(K.get_value(self.lr)),
                "epsilon": self.epsilon}


class Adadelta(Optimizer):
    '''Adadelta optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Learning rate. It is recommended to leave it at the default value.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.

    # References
        - [Adadelta - an adaptive learning rate method](http://arxiv.org/abs/1212.5701)
    '''
    def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6, *args, **kwargs):
        super(Adadelta, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = K.variable(lr)

    def get_updates(self, params, constraints, loss):
        grad_dict = self.get_gradients(loss, params)
        param_list = list(params.keys())
        accumulators = [K.variable(np.zeros(K.get_value(p).shape)) for p in param_list]
        delta_accumulators = [K.variable(np.zeros(K.get_value(p).shape)) for p in param_list]
        self.updates = []

        for p, a, d_a in zip(param_list, accumulators, delta_accumulators):
            # update accumulator
            new_a = tensor_set(a, grad_dict[p],
                               lambda x, y: self.rho * x + (1 - self.rho) * K.square(y))
            self.updates.append((a, new_a))

            # use the new accumulator and the *old* delta_accumulator
            # update = g * K.sqrt(d_a + self.epsilon) / K.sqrt(new_a + self.epsilon)

            new_p = tensor_set(
                p, grad_dict[p],
                lambda x, y, p, q: x - self.lr * (y * K.sqrt(p + self.epsilon) /
                                                  K.sqrt(q + self.epsilon)),
                d_a, new_a)
            self.updates.append((p, constraints[p](new_p)))  # apply constraints

            # update delta_accumulator
            new_d_a = tensor_set(
                d_a, grad_dict[p],
                lambda x, y, p, q: (self.rho * x + (1 - self.rho) *
                                    K.square(y * K.sqrt(p + self.epsilon) / K.sqrt(q + self.epsilon))),
                d_a, new_a)
            self.updates.append((d_a, new_d_a))
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(K.get_value(self.lr)),
                "rho": self.rho,
                "epsilon": self.epsilon}


class Adam(Optimizer):
    '''Adam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.

    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 *args, **kwargs):
        super(Adam, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)

    def get_updates(self, params, constraints, loss):
        grad_dict = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations+1.)]

        t = self.iterations + 1
        lr_t = self.lr * K.sqrt(1 - K.pow(self.beta_2, t)) / (1 - K.pow(self.beta_1, t))

        for p in grad_dict:
            # zero init of moment
            m = K.variable(np.zeros(K.get_value(p).shape))
            # zero init of velocity
            v = K.variable(np.zeros(K.get_value(p).shape))

            m_t = tensor_set(m, grad_dict[p],
                             lambda x, y: (self.beta_1 * x) + (1 - self.beta_1) * y)
            v_t = tensor_set(v, grad_dict[p],
                             lambda x, y: (self.beta_2 * x) + (1 - self.beta_2) * K.square(y))
            p_t = tensor_set(p, grad_dict[p],
                             lambda x, y, a, b: x - lr_t * a / (K.sqrt(b) + self.epsilon),
                             m_t, v_t)

            self.updates.append((m, m_t))
            self.updates.append((v, v_t))
            self.updates.append((p, constraints[p](p_t)))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(K.get_value(self.lr)),
                "beta_1": float(K.get_value(self.beta_1)),
                "beta_2": float(K.get_value(self.beta_2)),
                "epsilon": self.epsilon}


class Adamax(Optimizer):
    '''Adamax optimizer from Adam paper's Section 7. It is a variant
     of Adam based on the infinity norm.

    Default parameters follow those provided in the paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.

    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 *args, **kwargs):
        super(Adamax, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)

    def get_updates(self, params, constraints, loss):
        grad_dict = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations+1.)]

        t = self.iterations + 1
        lr_t = self.lr / (1 - K.pow(self.beta_1, t))

        for p in params:
            # zero init of 1st moment
            m = K.variable(np.zeros(K.get_value(p).shape))
            # zero init of exponentially weighted infinity norm
            u = K.variable(np.zeros(K.get_value(p).shape))

            m_t = tensor_set(m, grad_dict[p],
                             lambda x, y: (self.beta_1 * x) + (1 - self.beta_1) * y)
            u_t = tensor_set(u, grad_dict[p],
                             lambda x, y: K.maximum(self.beta_2 * x, K.abs(y)))
            p_t = tensor_set(p, grad_dict[p],
                             lambda x, y, a, b: x - lr_t * a / (b + self.epsilon),
                             m_t, u_t)

            self.updates.append((m, m_t))
            self.updates.append((u, u_t))
            self.updates.append((p, constraints[p](p_t)))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(K.get_value(self.lr)),
                "beta_1": float(K.get_value(self.beta_1)),
                "beta_2": float(K.get_value(self.beta_2)),
                "epsilon": self.epsilon}


# aliases
sgd = SGD
rmsprop = RMSprop
adagrad = Adagrad
adadelta = Adadelta
adam = Adam
adamax = Adamax


def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'optimizer',
                           instantiate=True, kwargs=kwargs)
