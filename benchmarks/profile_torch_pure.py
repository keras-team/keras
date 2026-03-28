import cProfile
import pstats

import torch
import torch.nn as tnn

dev = "mps"

cnn = (
    tnn.Sequential(
        tnn.Conv2d(3, 64, 3, padding=1),
        tnn.ReLU(),
        tnn.Conv2d(64, 64, 3, padding=1),
        tnn.ReLU(),
        tnn.MaxPool2d(2),
        tnn.Conv2d(64, 128, 3, padding=1),
        tnn.ReLU(),
        tnn.AdaptiveAvgPool2d(1),
        tnn.Flatten(),
        tnn.Linear(128, 10),
    )
    .to(dev)
    .eval()
)

imgs = torch.randn(4, 3, 32, 32, device=dev)
for _ in range(5):
    cnn(imgs)

pr = cProfile.Profile()
pr.enable()
for _ in range(200):
    cnn(imgs)
pr.disable()

p = pstats.Stats(pr)
p.strip_dirs().sort_stats("cumtime").print_stats(30)
