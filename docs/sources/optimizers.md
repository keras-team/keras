# Optimizers

## Base class

Optimizer

All optimizers descending from this class support the following keyword arguments:
- l1
- l2
- clipnorm
- maxnorm

##  SGD

SGD(lr=0.01, momentum=0., decay=0., nesterov=False)

##  Adagrad

Adagrad(lr=0.01, epsilon=1e-6)

##  Adadelta

Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)

##  RMSprop

RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)