import pytest

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Average
from keras.utils import np_utils
from keras.utils import test_utils
from keras import regularizers
from keras import backend as K

data_dim = 5
num_classes = 2
batch_size = 10


def get_data():
    (x_train, y_train), _ = test_utils.get_test_data(
        num_train=batch_size,
        num_test=batch_size,
        input_shape=(data_dim,),
        classification=True,
        num_classes=num_classes)
    y_train = np_utils.to_categorical(y_train, num_classes)

    return x_train, y_train


def create_model(kernel_regularizer=None, activity_regularizer=None):
    model = Sequential()
    model.add(Dense(num_classes,
                    kernel_regularizer=kernel_regularizer,
                    activity_regularizer=activity_regularizer,
                    input_shape=(data_dim,)))
    return model


def create_multi_input_model_from(layer1, layer2):
    input_1 = Input(shape=(data_dim,))
    input_2 = Input(shape=(data_dim,))
    out1 = layer1(input_1)
    out2 = layer2(input_2)
    out = Average()([out1, out2])
    model = Model([input_1, input_2], out)
    model.add_loss(K.mean(out2))
    model.add_loss(1)
    model.add_loss(1)
    return model


def test_kernel_regularization():
    x_train, y_train = get_data()
    for reg in [regularizers.l1(),
                regularizers.l2(),
                regularizers.l1_l2()]:
        model = create_model(kernel_regularizer=reg)
        model.compile(loss='categorical_crossentropy', optimizer='sgd')
        assert len(model.losses) == 1
        model.train_on_batch(x_train, y_train)


def test_activity_regularization():
    x_train, y_train = get_data()
    for reg in [regularizers.l1(), regularizers.l2()]:
        model = create_model(activity_regularizer=reg)
        model.compile(loss='categorical_crossentropy', optimizer='sgd')
        assert len(model.losses) == 1
        model.train_on_batch(x_train, y_train)


def test_regularization_shared_layer():
    dense_layer = Dense(num_classes,
                        kernel_regularizer=regularizers.l1(),
                        activity_regularizer=regularizers.l1())

    model = create_multi_input_model_from(dense_layer, dense_layer)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    assert len(model.losses) == 6


def test_regularization_shared_model():
    dense_layer = Dense(num_classes,
                        kernel_regularizer=regularizers.l1(),
                        activity_regularizer=regularizers.l1())

    input_tensor = Input(shape=(data_dim,))
    dummy_model = Model(input_tensor, dense_layer(input_tensor))

    model = create_multi_input_model_from(dummy_model, dummy_model)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    assert len(model.losses) == 6


def test_regularization_shared_layer_in_different_models():
    shared_dense = Dense(num_classes,
                         kernel_regularizer=regularizers.l1(),
                         activity_regularizer=regularizers.l1())
    models = []
    for _ in range(2):
        input_tensor = Input(shape=(data_dim,))
        unshared_dense = Dense(num_classes, kernel_regularizer=regularizers.l1())
        out = unshared_dense(shared_dense(input_tensor))
        models.append(Model(input_tensor, out))

    model = create_multi_input_model_from(*models)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    assert len(model.losses) == 8


if __name__ == '__main__':
    pytest.main([__file__])
