## Grapher

Creates a visualization of the model structure using `pydot`.

```python
grapher = keras.utils.dot_utils.Grapher()
```
- __Methods__:
    - __plot__(model, to_file): creates a graph visualizing the structure of `model` and writes it to `to_file`.
      - __Arguments__:
        - __model__: an instance of a Keras model (e.g. `Sequential`)
        - __to_file__: the filename to save the visualization png to.

__Examples__:

```python
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils.dot_utils import Grapher

grapher = Grapher()

model = Sequential()
model.add(Dense(64, 2, init='uniform'))
model.add(Activation('softmax'))
grapher.plot(model, 'model.png')
```

