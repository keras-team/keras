import numpy as np
from keras.visualization import charts
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


def test_get_model_parameter_flat():
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    assert len(charts.get_model_parameter_flat(model, 'W')) == 668672


def test_chart_config():
    chart_config = charts.ChartConfig()
    assert len(chart_config.charts) == 0
    chart_config.add_chart(charts.ParameterHistogram("W", title="Weights"))
    assert len(chart_config.charts) == 1
    chart_config = charts.ChartConfig._default_chartconfig(
                                    assert_no_chartconfig=False)
    assert len(chart_config.charts) > 0


def test_metric_linechart():
    chrt = charts.MetricLinechart(['loss'], ['val_loss'],
                                  batch_line_colors=['#53777A'],
                                  epoch_line_colors=['#C02942'])
    assert chrt.figure


def test_loss_linechart():
    chrt = charts.LossLinechart()
    assert chrt.figure


def test_acc_linechart():
    chrt = charts.AccuracyLinechart()
    assert chrt.figure


def test_parameter_linechart():
    chrt = charts.ParameterLinechart('W',
                                     transformer=np.mean,
                                     line_color='#C02942')
    assert chrt.figure


def test_parameter_histogram():
    chrt = charts.ParameterHistogram('W',
                                     bins=50, density=False,
                                     fill_alpha=0.6, fill_color='#D7455D',
                                     line_color='#C02942')
    assert chrt.figure
