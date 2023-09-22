from keras import optimizers
from keras.backend.torch.optimizers import torch_adam


class AdamW(torch_adam.Adam, optimizers.AdamW):
    pass
