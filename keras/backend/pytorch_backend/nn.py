from torch.nn import functional as F

from .random import random_binomial

def relu(x, alpha=0.0, max_value=None):
    if alpha != 0.0:
        negative_part = F.relu(-x)
    x = F.relu(x)
    if max_value is not None:
        x.clamp(0.0, max_value)
    if alpha != 0.0:
        x -= alpha * negative_part
    return x


def elu(x, alpha=1.0):
    return F.elu(x, alpha)


def softmax(x):
    return F.softmax(x)


def softplus(x):
    return F.softplus(x)


def softsign(x):
    return F.softsign(x)


def categorical_crossentropy(output, target, from_logits=False):
    if not from_logits:
        output /= output.sum(-1)
        output.clamp_(epsilon(), 1 - epsilon())
        return -(target * output.log()).sum(-1)
    else:
        output.clamp_(epsilon(), 1 - epsilon())
        return -output @ target + sum(list(map(exp, output))).log()


def sparse_categorical_crossentropy(output, target, from_logits=False):
    raise NotImplementedError


def binary_crossentropy(output, target, from_logits=False):
    if not from_logits:
        output.clamp_(epsilon(), 1 - epsilon())
        output = (output / (1 - output)).log()
    return y_true @ y_pred.log() - (1 - y_true) @ (1 - y_pred).log()


def sigmoid(x):
    return F.sigmoid(x)


def hard_sigmoid(x):
    x = (0.2 * x) + 0.5
    return x.clamp(x, 0, 1)


def tanh(x):
    return F.tanh(x)


def dropout(x, level, noise_shape=None, seed=None):
    assert 0.0 <= level <= 1.0
    if seed is not None:
        torch.manual_seed(sed)
    retain_prob = 1.0 - level
    dtype = x.numpy().dtype.name
    if noise_shape is None:
        random_tensor = random_binomial(x.size(), p=retain_prob, dtype=dtype)
    else:
        random_tensor = random_binomial(noise_shape, p=retain_prob, dtype=dtype)
        random_tesnor.expand_(x.size())
    x *= random_tensor
    x /= retain_prob
    return x


def l2_normalize(x, axis, epsilon=1e-12):
    square_sum = x.pow(2).sum(axis)
    norm = torch.max(square_sum, epsilon).pow(0.5)
    return x / norm


def in_top_k(predictions, targets, k):
    if k < 1:
        return torch.zeros(targets.size()).byte()

    if predictions.size()[1] <= k:
        return torch.ones(targets.size()).byte()

    predictions_k = torch.sort(predictions)[0][:, -k]
    targets_values = predictions[torch.arange(targets.size()[0]), targets]
    return predictions_k <= targets_values
