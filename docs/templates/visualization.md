
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
