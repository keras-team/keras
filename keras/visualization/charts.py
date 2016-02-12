from __future__ import print_function
from __future__ import division

import numpy as np
from keras import backend as K
from bokeh.io import gridplot
from bokeh.client import push_session
from bokeh.plotting import figure, curdoc


def get_model_parameter_flat(model, parameter):
    '''Utility method for accumulating all parameters of a given name
    across a model.

    Searches every module for a parameter and, if found, converts the
    parameter to a 1-d array (flattens it) by using numpy.ravel(), and
    appends it a global array.

    # Returns
        All values of a given parameter in a 1-d array.
    '''
    data = []

    for layer in model.layers:
        if hasattr(layer, parameter):
            param_value = K.eval(getattr(layer, parameter))
            data = list(param_value.ravel()) + data

    return np.array(data)

'''This is a remedy for when a chart_config is created but isn\'t specified
in the fit() method. Your chart will go horribly wrong if you don\'t declare
it as a keyword argument. Thus, whenever a config is first created, this
boolean is set. Setting of the default charts now depends on this being
False.
'''

config_created = False


class ChartConfig(object):
    '''Configuration for bokeh charts
    '''

    def __init__(self, cols=3, autoshow=True, session_id=None, url='default',
                 app_path='/'):
        '''Create a chart object.

        # Arguments
            cols: Number of columns in the chart interface. Default is 3.
            session_id: Session id to be passed to push_session()
            url: URL of your bokeh session
            app_path: directory where your app path is located
        '''

        self.charts = []
        self.cols = cols
        self.autoshow = autoshow
        self.session_id = session_id
        self.url = url
        self.app_path = app_path

        global config_created
        config_created = True

    def add_chart(self, chart):
        '''Add a chart to be rendered
        '''

        assert isinstance(chart, Chart)
        self.charts.append(chart)

    def finalize(self):
        '''Finalize the plot layout and (optionally) open the session
        in a webpage.
        '''

        self.session = push_session(curdoc(), session_id=self.session_id,
                                    url=self.url, app_path=self.app_path)

        rows = int(np.ceil(len(self.charts) / self.cols))
        formatted_charts = []
        for r in range(rows):
            formatted_charts.append([c.figure for c in self.charts[
                        r * self.cols:r * self.cols + self.cols]])
        layout = gridplot(formatted_charts)

        if self.autoshow:
            self.session.show(layout)

    def _publish_all(self, model, logs, event):
        '''Used in callback for event publishing to your charts.
        '''

        for chart in self.charts:
            chart.publish(model, logs, event)

    @staticmethod
    def _default_chartconfig(assert_no_chartconfig=True):
        '''Create a default chartconfig

        # Arguments
            assert_no_chartconfig: Used for testing
        '''

        if assert_no_chartconfig:
            global config_created
            assert(not config_created), ('Looks like you created a '
                                         'ChartConfig, but didn\'t specify it '
                                         'in your fit() method! Be sure to '
                                         'set chart_config=<your config> in '
                                         'fit().')

        chart_config = ChartConfig()
        chart_config.charts = [
            ParameterHistogram("W", title="Weights"),
            ParameterHistogram("b", title="Biases"),
            LossLinechart(title="Loss"),
            AccuracyLinechart(title="Accuracy")
        ]

        return chart_config


class Chart(object):
    '''Abstract base chart class.
    '''

    def __init__(self, title='Keras Chart', plot_width=350, plot_height=350,
                 tools="pan,wheel_zoom,box_zoom,reset,resize,save,hover",
                 background_fill_color="#FFFFFF", **kwargs):
        '''Create a new chart.

        Any arguments that are not explicitly stated here will be passed to
        the figure() initializer.

        # Arguments
            title: Title of the plot.
            plot_width: Width of the plot in pixels.
            plot_height: Height of the plot in pixels.
            tools: Tools to include in the bokeh plot.
            background_fill_color: Fill color for the background of the figure.
        '''

        self.figure = figure(title=title,
                             plot_width=plot_width,
                             plot_height=plot_height,
                             tools=tools,
                             background_fill_color=background_fill_color,
                             **kwargs)

    def publish(self, model, logs, event):
        '''Abstract method to publish events to your chart
        '''
        raise NotImplementedError("Must implement publish.")


class MetricLinechart(Chart):
    '''Chart that plots any metric provided by keras as a line chart.
    Some examples of metrics are \'loss\', \'val_loss\', \'acc\', and
    \'val_acc\'.
    '''

    def __init__(self, batch_metrics, epoch_metrics,
                 batch_line_colors=['#53777A'],
                 epoch_line_colors=['#C02942'], **kwargs):
        '''Create a new MetricLinechart.

        Any arguments that are not explicitly stated here will be passed to
        the figure() initializer.

        # Arguments
            batch_metrics: Metrics that are reported at the end of every batch.
                           The default functionality is to keep track of these
                           over every epoch then average them together. Notably
                           \'acc\' and \'loss\'. This argument is an array.
            epoch_metrics: Metrics that are reported at the end of every epoch.
                           Notably \'val_acc\' and \'loss\'. This argument is
                           an array.
            batch_line_colors: Corresponding colors for each batch_metric. Must
                               have at least enough colors to match each
                               metric. This argument is an array.
            epoch_line_colors: Corresponding colors for each epoch_metric. Must
                               have at least enough colors to match each
                               metric. This argument is an array.
        '''

        if not isinstance(batch_metrics, list):
            batch_metrics = [batch_metrics]

        if not isinstance(epoch_metrics, list):
            epoch_metrics = [epoch_metrics]

        if not isinstance(batch_line_colors, list):
            batch_line_colors = [batch_line_colors]

        if not isinstance(epoch_line_colors, list):
            epoch_line_colors = [epoch_line_colors]

        assert(len(batch_metrics) <= len(batch_line_colors)), (
            'Must have at least the same number of batch_colors '
            'as batch_metrics!')
        assert(len(epoch_metrics) <= len(epoch_line_colors)), (
            'Must have at least the same number of epoch_colors '
            'as epoch_metrics!')

        super(MetricLinechart, self).__init__(**kwargs)

        self.data = {}
        self.transient_data = {}
        self.lines = {}
        self.circles = {}
        self.batch_metrics = batch_metrics
        self.epoch_metrics = epoch_metrics

        for i, metric in enumerate(batch_metrics):
            self.lines[metric] = self.figure.line(
                [], [], line_color=batch_line_colors[i], legend=metric)
            self.circles[metric] = self.figure.circle(
                [], [], line_color=batch_line_colors[i],
                fill_color=batch_line_colors[i], legend=metric)

        for i, metric in enumerate(epoch_metrics):
            self.lines[metric] = self.figure.line(
                    [], [], line_color=epoch_line_colors[i], legend=metric)
            self.circles[metric] = self.figure.circle(
                    [], [], line_color=epoch_line_colors[i],
                    fill_color=epoch_line_colors[i], legend=metric)

    def publish(self, model, logs, event):
        '''Handling incoming events from Keras.
        '''

        if event == 'epoch_end':
            for metric in self.epoch_metrics:
                if metric not in self.data:
                    self.data[metric] = []

                self.data[metric].append(logs[metric])

            for metric in self.batch_metrics:
                if metric not in self.data:
                    self.data[metric] = []

                self.data[metric].append(np.mean(self.transient_data[metric]))

            self.transient_data = {}
            self.update_linechart()

        elif event == 'batch_end':
            for metric in self.batch_metrics:
                if metric not in self.transient_data:
                    self.transient_data[metric] = []

                self.transient_data[metric].append(logs[metric])

    def update_linechart(self):
        '''Utility method for propagating changes to the server.
        '''

        for metric in self.epoch_metrics:
            x = np.arange(1, len(self.data[metric]) + 1)
            y = self.data[metric]
            self.lines[metric].data_source.data.update(x=x, y=y)
            self.circles[metric].data_source.data.update(x=x, y=y)

        for metric in self.batch_metrics:
            x = np.arange(1, len(self.data[metric]) + 1)
            y = self.data[metric]
            self.lines[metric].data_source.data.update(x=x, y=y)
            self.circles[metric].data_source.data.update(x=x, y=y)


class LossLinechart(MetricLinechart):
    '''Prepackaged chart for plotting \'loss\' and \'val_loss\'.
    '''

    def __init__(self, show_loss=True, show_val_loss=True, **kwargs):
        batch_metrics = []
        epoch_metrics = []

        if show_loss:
            batch_metrics = ['loss']

        if show_val_loss:
            epoch_metrics = ['val_loss']

        super(LossLinechart, self).__init__(batch_metrics,
                                            epoch_metrics, **kwargs)


class AccuracyLinechart(MetricLinechart):
    '''Prepackaged chart for plotting \'acc\' and \'val_acc\'.
    '''

    def __init__(self, show_acc=True, show_val_acc=True, **kwargs):
        batch_metrics = []
        epoch_metrics = []

        if show_acc:
            batch_metrics = ['acc']

        if show_val_acc:
            epoch_metrics = ['val_acc']

        super(AccuracyLinechart, self).__init__(batch_metrics,
                                                epoch_metrics, **kwargs)
        self.figure.legend.location = 'bottom_right'


class ParameterLinechart(Chart):
    '''Chart that plots any parameter in your model as a line chart.
    Some examples of metrics are \'W\' or 'b' for the general name of the
    variables holding weights and biases for Dense layers.
    '''

    def __init__(self, parameter, transformer,
                 line_color='#C02942', **kwargs):
        '''Create a new ParameterLinechart.

        Any arguments that are not explicitly stated here will be passed to
        the figure() initializer.

        # Arguments
            parameter: Parameter to be watched across the layers. An example
                       would be \'W\', which contains the weights in Dense
                       layers.
            transformer: Function to apply to the list of values. The default
                         is to compute the mean, but you could also use
                         numpy.std if you wanted to plot the std of the
                         weights, for instance.
            line_color: Color of the line in the chart.
        '''

        super(ParameterLinechart, self).__init__(**kwargs)

        self.data = []
        self.parameter = parameter
        self.transformer = transformer
        self.r = self.figure.line([], [], line_color=line_color)
        self.r2 = self.figure.circle([], [], line_color=line_color)
        self.ds = self.r.data_source
        self.ds2 = self.r2.data_source

    def publish(self, model, logs, event):
        '''Handling incoming events from Keras.
        '''

        if event == 'epoch_end':
            self.data.append(self.transformer(
                get_model_parameter_flat(model, self.parameter)))
            self.update_linechart()

    def update_linechart(self):
        '''Utility method for propagating changes to the server.
        '''

        self.ds.data.update(y=self.data, x=np.arange(1, len(self.data) + 1))
        self.ds2.data.update(y=self.data, x=np.arange(1, len(self.data) + 1))


class ParameterHistogram(Chart):
    '''Chart that plots any parameter in your model as a histogram.
    Some examples of metrics are \'W\' or 'b' for the general name of the
    variables holding weights and biases for Dense layers.
    '''

    def __init__(self, parameter,
                 bins=50, density=False,
                 fill_alpha=0.6, fill_color='#D7455D',
                 line_color='#C02942',  **kwargs):
        '''Create a new ParameterHistogram.

        Any arguments that are not explicitly stated here will be passed to
        the figure() initializer.

        # Arguments
            parameter: Parameter to be watched across the layers. An example
                       would be \'W\', which contains the weights in Dense
                       layers.
            bins: Number of bins in the histogram
            density: Compute histogram as a probability density.
            fill_alpha: Alpha parameter (opacity) when filling elements in the
                        chart.
            fill_color: Color to use when filling elements in the chart.
            line_color: Color to use when drawing the borders of the histogram.
        '''

        super(ParameterHistogram, self).__init__(**kwargs)
        self.data = []
        self.parameter = parameter
        self.bins = bins
        self.density = density

        self.r = self.figure.quad(top=np.zeros(self.bins),
                                  bottom=np.zeros(self.bins),
                                  left=np.zeros(self.bins + 1)[:-1],
                                  right=np.zeros(self.bins + 1)[1:],
                                  fill_color=fill_color,
                                  line_color=line_color,
                                  fill_alpha=fill_alpha)
        self.ds = self.r.data_source

    def publish(self, model, logs, event):
        '''Handling incoming events from Keras.
        '''

        if event == 'epoch_end':
            self.data = get_model_parameter_flat(model, self.parameter)
            self.update_histogram()

    def update_histogram(self):
        '''Utility method for propagating changes to the server.
        '''

        hist, edges = np.histogram(self.data,
                                   density=self.density,
                                   bins=self.bins)
        self.ds.data.update(top=hist,
                            left=edges[:-1],
                            right=edges[1:])
