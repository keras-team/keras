# Regularizers

## Usage of regularizers

Regularizers allow the use of penalties on particular sets of parameters during optimization.

A penalty is initilized with its weight during optimization: `l1(.05)`
The keyword arguments used for passing penalties to parameters in a layer will depend on the layer. 
For weights in the `Dense` layer it is simply `W_regularizer`
For biases in the `Dense` layer it is simply `b_regularizer`

```python
model.add(Dense(64, 64, W_regularizer = l2(.01)))
```

## Available penalties

- __l1__: L1 regularization penalty, also known as LASSO
- __l2__: L2 regularization penalty, also known as weight decay
