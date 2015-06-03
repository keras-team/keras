
## Usage of optimizers

An optimizer is one of the two arguments required for compiling a Keras model:

```python
model = Sequential()
model.add(Dense(20, 64, init='uniform'))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

You can either instantiate an optimizer before passing it to `model.compile()` , as in the above example, or you can call it by its name. In the latter case, the default parameters for the optimizer will be used.

```python
# pass optimizer by name: default parameters will be used
model.compile(loss='mean_squared_error', optimizer='sgd')
```

---

## Base class

```python
keras.optimizers.Optimizer(**kwargs)
```

All optimizers descended from this class support the following keyword argument:

- __clipnorm__: float >= 0.

Note: this is base class for building optimizers, not an actual optimizer that can be used for training models.

---

##  SGD

```python
keras.optimizers.SGD(lr=0.01, momentum=0., decay=0., nesterov=False)
``` 

__Arguments__:

- __lr__: float >= 0. Learning rate.
- __momentum__: float >= 0. Parameter updates momentum.
- __decay__: float >= 0. Learning rate decay over each update.
- __nesterov__: boolean. Whether to apply Nesterov momentum.

---

##  Adagrad

```python
keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6)
```

It is recommended to leave the parameters of this optimizer at their default values.

__Arguments__:

- __lr__: float >= 0. Learning rate. 
- __epsilon__: float >= 0. 

---

##  Adadelta

```python
keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
```

It is recommended to leave the parameters of this optimizer at their default values.

__Arguments__:

- __lr__: float >= 0. Learning rate. It is recommended to leave it at the default value.
- __rho__: float >= 0. 
- __epsilon__: float >= 0. Fuzz factor.

For more info, see *"Adadelta: an adaptive learning rate method"* by Matthew Zeiler.

---

##  RMSprop 

```python
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
```

It is recommended to leave the parameters of this optimizer at their default values.

__Arguments__:

- __lr__: float >= 0. Learning rate. 
- __rho__: float >= 0.
- __epsilon__: float >= 0. Fuzz factor.

---

## Adam

```python
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, kappa=1-1e-8)
```

Adam optimizer, proposed by Kingma and Lei Ba in [Adam: A Method For Stochastic Optimization](http://arxiv.org/pdf/1412.6980v4.pdf). Default parameters are those suggested in the paper. The parameter "lambda" from the paper has been renamed kappa, for syntactic reasons.

__Arguments__:

- __lr__: float >= 0. Learning rate. 
- __beta_1__, __beta_2__: floats, 0 < beta < 1. Generally close to 1.
- __epsilon__: float >= 0. Fuzz factor.
- __kappa__: float 0 < kappa < 1. Lambda parameter in the original paper.

---