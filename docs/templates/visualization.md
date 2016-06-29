
## Model visualization

The `keras.utils.visualize_util` module provides utility functions to plot
a Keras model (using graphviz).

This will plot a graph of the model and save it to a file:
```python
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')
```

`plot` takes two optional arguments:

- `show_shapes` (defaults to False) controls whether output shapes are shown in the graph.
- `show_layer_names` (defaults to True) controls whether layer names are shown in the graph.

You can also directly obtain the `pydot.Graph` object and render it yourself,
for example to show it in an ipython notebook :
```python
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```
