
## Model visualization

The `keras.utils.vis_utils` module provides utility functions to plot
a Keras model (using `graphviz`).

This will plot a graph of the model and save it to a file:
```python
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

`plot_model` takes two optional arguments:

- `show_shapes` (defaults to False) controls whether output shapes are shown in the graph.
- `show_layer_names` (defaults to True) controls whether layer names are shown in the graph.

You can also directly obtain the `pydot.Graph` object and render it yourself,
for example to show it in an ipython notebook :
```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```

## Training history visualization

The `keras.model.fit` method returns a `History` object. The `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable). Using the `matplotlib` library, this `history` attribute can be plotted. For example:

```python
history = model.fit(X, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

import matplotlib.pyplot as plt

# Plot history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

```
