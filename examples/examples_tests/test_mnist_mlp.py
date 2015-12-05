from __future__ import absolute_import
from __future__ import print_function


from examples.mnist_mlp import run_mnist_mlp


def test_mnist_mlp_example():
    score = run_mnist_mlp(nb_epoch=1)
    # loose loss bounds just to check that mlp is not completely off
    assert(score[1] > 0.95)


if __name__ == '__main__':
    test_mnist_mlp_example()
