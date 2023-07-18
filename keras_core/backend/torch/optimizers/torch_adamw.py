from keras_core import optimizers
from keras_core.backend.torch.optimizers import torch_adam


class AdamW(torch_adam.Adam, optimizers.AdamW):
    pass
