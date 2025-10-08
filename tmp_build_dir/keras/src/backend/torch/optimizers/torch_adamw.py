from keras.src import optimizers
from keras.src.backend.torch.optimizers import torch_adam


class AdamW(torch_adam.Adam, optimizers.AdamW):
    pass
