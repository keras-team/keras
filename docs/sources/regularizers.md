## Usage of regularizers

Regularizers allow to apply penalties on network parameters during optimization.

The keyword arguments used for passing penalties to parameters in a layer will depend on the layer. 

In the `Dense` layer it is simply `W_regularizer` for the main weights matrix, and `b_regularizer` for the bias.

```python
from keras.regularizers import l2
model.add(Dense(64, 64, W_regularizer = l2(.01)))
```

## Available penalties

- __l1__(l=0.01): L1 regularization penalty, also known as LASSO
- __l2__(l=0.01): L2 regularization penalty, also known as weight decay, or Ridge
