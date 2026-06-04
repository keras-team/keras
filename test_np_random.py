import numpy as np
seed = 3481
np.random.seed(seed)
inputs = np.random.randint(0, 255, size=(224, 224, 3))
print(inputs.shape)
