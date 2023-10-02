import torch

from keras import layers
from keras import testing
from keras.backend.common import KerasVariable


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = layers.Dense(1)

    def forward(self, x):
        x = self.fc1(x)
        return x


class TorchWorkflowTest(testing.TestCase):
    def test_keras_layer_in_nn_module(self):
        net = Net()

        # Test using Keras layer in a nn.Module.
        # Test forward pass
        self.assertAllEqual(list(net(torch.empty(100, 10)).shape), [100, 1])
        # Test KerasVariables are added as nn.Parameter.
        self.assertLen(list(net.parameters()), 2)

        # Test using KerasVariable as a torch tensor for torch ops.
        kernel = net.fc1.kernel
        transposed_kernel = torch.transpose(kernel, 0, 1)
        self.assertIsInstance(kernel, KerasVariable)
        self.assertIsInstance(
            torch.mul(kernel, transposed_kernel), torch.Tensor
        )
