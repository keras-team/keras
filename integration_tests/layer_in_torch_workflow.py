import torch

from keras_core import layers


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = layers.Dense(1)

    def forward(self, x):
        x = self.fc1(x)
        return x


net = Net()

assert list(net(torch.empty(100, 10)).shape) == [100, 1]
assert len(list(net.parameters())) == 2
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
