import numpy as np
from scipy import linalg


def cholesky(a):
    return np.linalg.cholesky(a)


def det(a):
    return np.linalg.det(a)


def eig(a):
    return np.linalg.eig(a)


def inv(a):
    return np.linalg.inv(a)


def solve(a, b):
    return np.linalg.solve(a, b)
