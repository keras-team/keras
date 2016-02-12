
## Static model visualization

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

## Real-time visualization

Keras also provides a library for creating interactive visualizations
built on [Bokeh][1].

**Note**: Since this technology is built on top of Bokeh, you should run
Bokeh's server in the background. You can install Bokeh and start the server
by

```bash
pip install bokeh
bokeh serve
```

Testing with the default charts is as easy as adding `visualize=True` to your
fit method. You can create custom visualizations by using
`keras.visualization.charts.ChartConfig`. For example

```python
from keras.visualization import charts
chart_config = charts.ChartConfig()
chart_config.add_chart(charts.ParameterHistogram('W', title='Weights'))
chart_config.add_chart(charts.ParameterHistogram('b', title='Biases'))
chart_config.add_chart(charts.ParameterHistogram('alphas', title='Alphas'))
chart_config.add_chart(charts.LossLinechart(title='Loss'))
chart_config.add_chart(charts.AccuracyLinechart(title='Accuracy'))

model.fit(X_train, Y_train, batch_size=128, nb_epoch=50,
          show_accuracy=True, verbose=2, validation_data=(X_test, Y_test),
          visualize=True, chart_config=chart_config)
```

### Custom charts

Creating your own charts is just as easy --- simply inherit from the
`keras.visualization.charts.Chart` class and you have everything you need to
create interactive graphs in Keras! Some important things to note:

1. **ALWAYS** call `super(<yourclass>, self).__init__(**kwargs)`. This ensures
that the figures are set up correctly.
2. Everything about your graph will be accessible through `figure`, which is
just a normal Bokeh figure object.
3. Your chart receives updates by implementing the `publish` method. Possible
values are any callback from the [callbacks](../callbacks/) module.

An example chart that plots a 1 for an epoch starting and 0 for an epoch
ending.

```python
class NewChart(charts.Chart):
    def __init__(self, **kwargs):
        super(NewChart, self).__init__(**kwargs)
        self.data = []
        self.line_graph = self.figure.line([], [])

    def publish(self, model, logs, event):
        if event == 'epoch_begin':
            self.data.append(1)
        elif event == 'epoch_end':
            self.data.append(0)

        self.line_graph.data_source.data.update(x=np.arange(len(self.data)),
                                                y=self.data)
```

[1]: http://bokeh.pydata.org/
