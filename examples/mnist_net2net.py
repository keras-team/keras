'''This is an implementation of Net2Net experiment with MNIST in
'Net2Net: Accelerating Learning via Knowledge Transfer'
by Tianqi Chen, Ian Goodfellow, and Jonathon Shlens

arXiv:1511.05641v4 [cs.LG] 23 Apr 2016
http://arxiv.org/abs/1511.05641

# Notes

- What:
  + Net2Net is a group of methods to transfer knowledge from a teacher neural
    net to a student net,so that the student net can be trained faster than
    from scratch.
  + The paper discussed two specific methods of Net2Net, i.e. Net2WiderNet
    and Net2DeeperNet.
  + Net2WiderNet replaces a model with an equivalent wider model that has
    more units in each hidden layer.
  + Net2DeeperNet replaces a model with an equivalent deeper model.
  + Both are based on the idea of 'function-preserving transformations of
    neural nets'.
- Why:
  + Enable fast exploration of multiple neural nets in experimentation and
    design process,by creating a series of wider and deeper models with
    transferable knowledge.
  + Enable 'lifelong learning system' by gradually adjusting model complexity
    to data availability,and reusing transferable knowledge.

# Experiments

- Teacher model: a basic CNN model trained on MNIST for 3 epochs.
- Net2WiderNet experiment:
  + Student model has a wider Conv2D layer and a wider FC layer.
  + Comparison of 'random-padding' vs 'net2wider' weight initialization.
  + With both methods, after 1 epoch, student model should perform as well as
    teacher model, but 'net2wider' is slightly better.
- Net2DeeperNet experiment:
  + Student model has an extra Conv2D layer and an extra FC layer.
  + Comparison of 'random-init' vs 'net2deeper' weight initialization.
  + After 1 epoch, performance of 'net2deeper' is better than 'random-init'.
- Hyper-parameters:
  + SGD with momentum=0.9 is used for training teacher and student models.
  + Learning rate adjustment: it's suggested to reduce learning rate
    to 1/10 for student model.
  + Addition of noise in 'net2wider' is used to break weight symmetry
    and thus enable full capacity of student models. It is optional
    when a Dropout layer is used.

# Results

- Tested with TF backend and 'channels_last' image_data_format.
- Running on GPU GeForce GTX Titan X Maxwell
- Performance Comparisons - validation loss values during first 3 epochs:

Teacher model ...
(0) teacher_model:             0.0537   0.0354   0.0356

Experiment of Net2WiderNet ...
(1) wider_random_pad:          0.0320   0.0317   0.0289
(2) wider_net2wider:           0.0271   0.0274   0.0270

Experiment of Net2DeeperNet ...
(3) deeper_random_init:        0.0682   0.0506   0.0468
(4) deeper_net2deeper:         0.0292   0.0294   0.0286
'''

from __future__ import print_function
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import SGD
from keras.datasets import mnist

if K.image_data_format() == 'channels_first':
    input_shape = (1, 28, 28)  # image shape
else:
    input_shape = (28, 28, 1)  # image shape
num_classes = 10  # number of classes
epochs = 3


# load and pre-process data
def preprocess_input(x):
    return x.astype('float32').reshape((-1,) + input_shape) / 255


def preprocess_output(y):
    return keras.utils.to_categorical(y)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = map(preprocess_input, [x_train, x_test])
y_train, y_test = map(preprocess_output, [y_train, y_test])
print('Loading MNIST data...')
print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape, 'y_test shape', y_test.shape)


# knowledge transfer algorithms
def wider2net_conv2d(teacher_w1, teacher_b1, teacher_w2, new_width, init):
    '''Get initial weights for a wider conv2d layer with a bigger filters,
    by 'random-padding' or 'net2wider'.

    # Arguments
        teacher_w1: `weight` of conv2d layer to become wider,
          of shape (filters1, num_channel1, kh1, kw1)
        teacher_b1: `bias` of conv2d layer to become wider,
          of shape (filters1, )
        teacher_w2: `weight` of next connected conv2d layer,
          of shape (filters2, num_channel2, kh2, kw2)
        new_width: new `filters` for the wider conv2d layer
        init: initialization algorithm for new weights,
          either 'random-pad' or 'net2wider'
    '''
    assert teacher_w1.shape[0] == teacher_w2.shape[1], (
        'successive layers from teacher model should have compatible shapes')
    assert teacher_w1.shape[3] == teacher_b1.shape[0], (
        'weight and bias from same layer should have compatible shapes')
    assert new_width > teacher_w1.shape[3], (
        'new width (filters) should be bigger than the existing one')

    n = new_width - teacher_w1.shape[3]
    if init == 'random-pad':
        new_w1 = np.random.normal(0, 0.1, size=teacher_w1.shape[:3] + (n,))
        new_b1 = np.ones(n) * 0.1
        new_w2 = np.random.normal(
            0, 0.1,
            size=teacher_w2.shape[:2] + (n, teacher_w2.shape[3]))
    elif init == 'net2wider':
        index = np.random.randint(teacher_w1.shape[3], size=n)
        factors = np.bincount(index)[index] + 1.
        new_w1 = teacher_w1[:, :, :, index]
        new_b1 = teacher_b1[index]
        new_w2 = teacher_w2[:, :, index, :] / factors.reshape((1, 1, -1, 1))
    else:
        raise ValueError('Unsupported weight initializer: %s' % init)

    student_w1 = np.concatenate((teacher_w1, new_w1), axis=3)
    if init == 'random-pad':
        student_w2 = np.concatenate((teacher_w2, new_w2), axis=2)
    elif init == 'net2wider':
        # add small noise to break symmetry, so that student model will have
        # full capacity later
        noise = np.random.normal(0, 5e-2 * new_w2.std(), size=new_w2.shape)
        student_w2 = np.concatenate((teacher_w2, new_w2 + noise), axis=2)
        student_w2[:, :, index, :] = new_w2
    student_b1 = np.concatenate((teacher_b1, new_b1), axis=0)

    return student_w1, student_b1, student_w2


def wider2net_fc(teacher_w1, teacher_b1, teacher_w2, new_width, init):
    '''Get initial weights for a wider fully connected (dense) layer
       with a bigger nout, by 'random-padding' or 'net2wider'.

    # Arguments
        teacher_w1: `weight` of fc layer to become wider,
          of shape (nin1, nout1)
        teacher_b1: `bias` of fc layer to become wider,
          of shape (nout1, )
        teacher_w2: `weight` of next connected fc layer,
          of shape (nin2, nout2)
        new_width: new `nout` for the wider fc layer
        init: initialization algorithm for new weights,
          either 'random-pad' or 'net2wider'
    '''
    assert teacher_w1.shape[1] == teacher_w2.shape[0], (
        'successive layers from teacher model should have compatible shapes')
    assert teacher_w1.shape[1] == teacher_b1.shape[0], (
        'weight and bias from same layer should have compatible shapes')
    assert new_width > teacher_w1.shape[1], (
        'new width (nout) should be bigger than the existing one')

    n = new_width - teacher_w1.shape[1]
    if init == 'random-pad':
        new_w1 = np.random.normal(0, 0.1, size=(teacher_w1.shape[0], n))
        new_b1 = np.ones(n) * 0.1
        new_w2 = np.random.normal(0, 0.1, size=(n, teacher_w2.shape[1]))
    elif init == 'net2wider':
        index = np.random.randint(teacher_w1.shape[1], size=n)
        factors = np.bincount(index)[index] + 1.
        new_w1 = teacher_w1[:, index]
        new_b1 = teacher_b1[index]
        new_w2 = teacher_w2[index, :] / factors[:, np.newaxis]
    else:
        raise ValueError('Unsupported weight initializer: %s' % init)

    student_w1 = np.concatenate((teacher_w1, new_w1), axis=1)
    if init == 'random-pad':
        student_w2 = np.concatenate((teacher_w2, new_w2), axis=0)
    elif init == 'net2wider':
        # add small noise to break symmetry, so that student model will have
        # full capacity later
        noise = np.random.normal(0, 5e-2 * new_w2.std(), size=new_w2.shape)
        student_w2 = np.concatenate((teacher_w2, new_w2 + noise), axis=0)
        student_w2[index, :] = new_w2
    student_b1 = np.concatenate((teacher_b1, new_b1), axis=0)

    return student_w1, student_b1, student_w2


def deeper2net_conv2d(teacher_w):
    '''Get initial weights for a deeper conv2d layer by net2deeper'.

    # Arguments
        teacher_w: `weight` of previous conv2d layer,
          of shape (kh, kw, num_channel, filters)
    '''
    kh, kw, num_channel, filters = teacher_w.shape
    student_w = np.zeros_like(teacher_w)
    for i in range(filters):
        student_w[(kh - 1) // 2, (kw - 1) // 2, i, i] = 1.
    student_b = np.zeros(filters)
    return student_w, student_b


def copy_weights(teacher_model, student_model, layer_names):
    '''Copy weights from teacher_model to student_model,
     for layers with names listed in layer_names
    '''
    for name in layer_names:
        weights = teacher_model.get_layer(name=name).get_weights()
        student_model.get_layer(name=name).set_weights(weights)


# methods to construct teacher_model and student_models
def make_teacher_model(x_train, y_train,
                       x_test, y_test,
                       epochs):
    '''Train and benchmark performance of a simple CNN.
    (0) Teacher model
    '''
    model = Sequential()
    model.add(Conv2D(64, 3, input_shape=input_shape,
                     padding='same', name='conv1'))
    model.add(MaxPooling2D(2, name='pool1'))
    model.add(Conv2D(64, 3, padding='same', name='conv2'))
    model.add(MaxPooling2D(2, name='pool2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(64, activation='relu', name='fc1'))
    model.add(Dense(num_classes, activation='softmax', name='fc2'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, momentum=0.9),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=epochs,
              validation_data=(x_test, y_test))
    return model


def make_wider_student_model(teacher_model,
                             x_train, y_train,
                             x_test, y_test,
                             init, epochs):
    '''Train a wider student model based on teacher_model,
       with either 'random-pad' (baseline) or 'net2wider'
    '''
    new_conv1_width = 128
    new_fc1_width = 128

    model = Sequential()
    # a wider conv1 compared to teacher_model
    model.add(Conv2D(new_conv1_width, 3, input_shape=input_shape,
                     padding='same', name='conv1'))
    model.add(MaxPooling2D(2, name='pool1'))
    model.add(Conv2D(64, 3, padding='same', name='conv2'))
    model.add(MaxPooling2D(2, name='pool2'))
    model.add(Flatten(name='flatten'))
    # a wider fc1 compared to teacher model
    model.add(Dense(new_fc1_width, activation='relu', name='fc1'))
    model.add(Dense(num_classes, activation='softmax', name='fc2'))

    # The weights for other layers need to be copied from teacher_model
    # to student_model, except for widened layers
    # and their immediate downstreams, which will be initialized separately.
    # For this example there are no other layers that need to be copied.

    w_conv1, b_conv1 = teacher_model.get_layer('conv1').get_weights()
    w_conv2, b_conv2 = teacher_model.get_layer('conv2').get_weights()
    new_w_conv1, new_b_conv1, new_w_conv2 = wider2net_conv2d(
        w_conv1, b_conv1, w_conv2, new_conv1_width, init)
    model.get_layer('conv1').set_weights([new_w_conv1, new_b_conv1])
    model.get_layer('conv2').set_weights([new_w_conv2, b_conv2])

    w_fc1, b_fc1 = teacher_model.get_layer('fc1').get_weights()
    w_fc2, b_fc2 = teacher_model.get_layer('fc2').get_weights()
    new_w_fc1, new_b_fc1, new_w_fc2 = wider2net_fc(
        w_fc1, b_fc1, w_fc2, new_fc1_width, init)
    model.get_layer('fc1').set_weights([new_w_fc1, new_b_fc1])
    model.get_layer('fc2').set_weights([new_w_fc2, b_fc2])

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.001, momentum=0.9),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=epochs,
              validation_data=(x_test, y_test))


def make_deeper_student_model(teacher_model,
                              x_train, y_train,
                              x_test, y_test,
                              init, epochs):
    '''Train a deeper student model based on teacher_model,
       with either 'random-init' (baseline) or 'net2deeper'
    '''
    model = Sequential()
    model.add(Conv2D(64, 3, input_shape=input_shape,
                     padding='same', name='conv1'))
    model.add(MaxPooling2D(2, name='pool1'))
    model.add(Conv2D(64, 3, padding='same', name='conv2'))
    # add another conv2d layer to make original conv2 deeper
    if init == 'net2deeper':
        prev_w, _ = model.get_layer('conv2').get_weights()
        new_weights = deeper2net_conv2d(prev_w)
        model.add(Conv2D(64, 3, padding='same',
                         name='conv2-deeper', weights=new_weights))
    elif init == 'random-init':
        model.add(Conv2D(64, 3, padding='same', name='conv2-deeper'))
    else:
        raise ValueError('Unsupported weight initializer: %s' % init)
    model.add(MaxPooling2D(2, name='pool2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(64, activation='relu', name='fc1'))
    # add another fc layer to make original fc1 deeper
    if init == 'net2deeper':
        # net2deeper for fc layer with relu, is just an identity initializer
        model.add(Dense(64, kernel_initializer='identity',
                        activation='relu', name='fc1-deeper'))
    elif init == 'random-init':
        model.add(Dense(64, activation='relu', name='fc1-deeper'))
    else:
        raise ValueError('Unsupported weight initializer: %s' % init)
    model.add(Dense(num_classes, activation='softmax', name='fc2'))

    # copy weights for other layers
    copy_weights(teacher_model, model, layer_names=[
                 'conv1', 'conv2', 'fc1', 'fc2'])

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.001, momentum=0.9),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=epochs,
              validation_data=(x_test, y_test))


# experiments setup
def net2wider_experiment():
    '''Benchmark performances of
    (1) a wider student model with `random_pad` initializer
    (2) a wider student model with `Net2WiderNet` initializer
    '''
    print('\nExperiment of Net2WiderNet ...')

    print('\n(1) building wider student model by random padding ...')
    make_wider_student_model(teacher_model,
                             x_train, y_train,
                             x_test, y_test,
                             init='random-pad',
                             epochs=epochs)
    print('\n(2) building wider student model by net2wider ...')
    make_wider_student_model(teacher_model,
                             x_train, y_train,
                             x_test, y_test,
                             init='net2wider',
                             epochs=epochs)


def net2deeper_experiment():
    '''Benchmark performances of
    (3) a deeper student model with `random_init` initializer
    (4) a deeper student model with `Net2DeeperNet` initializer
    '''
    print('\nExperiment of Net2DeeperNet ...')

    print('\n(3) building deeper student model by random init ...')
    make_deeper_student_model(teacher_model,
                              x_train, y_train,
                              x_test, y_test,
                              init='random-init',
                              epochs=epochs)
    print('\n(4) building deeper student model by net2deeper ...')
    make_deeper_student_model(teacher_model,
                              x_train, y_train,
                              x_test, y_test,
                              init='net2deeper',
                              epochs=epochs)


print('\n(0) building teacher model ...')
teacher_model = make_teacher_model(x_train, y_train,
                                   x_test, y_test,
                                   epochs=epochs)

# run the experiments
net2wider_experiment()
net2deeper_experiment()
