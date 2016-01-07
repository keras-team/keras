
## Model visualization

The `keras.utils.visualize_util` module provides utility functions to plot
a Keras model (using graphviz).

This will plot a graph of the model and save it to a file:
```python
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')
```

You can also directly obtain the `pydot.Graph` object and render it yourself,
for example to show it in an ipython notebook :
```python
from IPython.display import SVG
from keras.utils.visualize_util import to_graph

SVG(to_graph(model).create(prog='dot', format='svg'))
```
