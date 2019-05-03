"""
This example trains a simple CNN-Capsule Network on the CIFAR10 data set.

Without Data Augmentation:
It gets to 75% validation accuracy in 10 epochs, 79% after 15 epochs,
and overfitting after 20 epochs

With Data Augmentation:
It gets to 75% validation accuracy in 10 epochs, 79% after 15 epochs,
and 83% after 30 epochs.

The highest achieved validation accuracy is 83.79% after 50 epochs.
This is a fast implementation that takes just 20s/epoch on a GTX 1070 GPU.

The paper "Dynamic Routing Between Capsules": https://arxiv.org/abs/1710.09829
"""
from __future__ import print_function

from keras import activations
from keras import backend as K
from keras import layers
from keras import utils
from keras.datasets import cifar10
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


def squash(x, axis=-1):
    """The Squashing Function.
    The nonlinear activation function used in Capsule Network
    # Arguments
        x: Input Tensor.
        axis: Integer axis along which the squashing function is to be applied.

    # Returns
        Tensor with scaled value of the input tensor
    """
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


def margin_loss(y_true, y_pred):
    """Margin loss

    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.

    # Returns
        Tensor with one scalar loss entry per sample.
    """
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)


class Capsule(layers.Layer):
    """Capsule Network

    A Capsule Network Layer implementation in Keras
    There are two versions of Capsule Networks.
    One is similar to dense layer (for the fixed-shape input),
    and the other is similar to time distributed dense layer
    (for inputs of varied length).

    The input shape of Capsule must be (batch_size,
                                        input_num_capsule,
                                        input_dim_capsule
                                       )
    and the output shape is (batch_size,
                             num_capsule,
                             dim_capsule
                            )
    The Capsule implementation is from https://github.com/bojone/Capsule/


    # Arguments
        num_capsule: An integer, the number of capsules.
        dim_capsule: An integer, the dimensions of the capsule.
        routings: An integer, the number of routings.
        share_weights: A boolean, sets weight sharing between layers.
        activation: A string, the activation function to be applied.
    """

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs, **kwargs):
        """Following the routing algorithm from Hinton's paper,
        but replace b = b + <u,v> with b = <u,v>.

        This change can improve the feature representation of the capsule.

        However, you can replace
            b = K.batch_dot(outputs, hat_inputs, [2, 3])
        with
            b += K.batch_dot(outputs, hat_inputs, [2, 3])
        to get standard routing.
        """

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        print(self.routings)
        for i in range(self.routings):
            c = K.softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)
        return o

    def compute_output_shape(self, input_shape):
        return None, self.num_capsule, self.dim_capsule


batch_size = 128
num_classes = 10
epochs = 100
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# A simple Conv2D model
input_image = layers.Input(shape=(None, None, 3))
x = layers.Conv2D(64, (3, 3), activation='relu')(input_image)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.AveragePooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)

# Now, we reshape it to (batch_size, input_num_capsule, input_dim_capsule)
# then connect a capsule layer.
# The output of final model is the lengths of 10 capsules, which have 16 dimensions.
# The length of the output vector of the capsule expresses the probability of
# existence of the entity, so the problem becomes a 10 two-classification problem.

x = layers.Reshape((-1, 128))(x)
capsule = Capsule(10, 16, 3, True)(x)
output = layers.Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
model = Model(inputs=input_image, outputs=output)

# Margin loss is used
model.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])
model.summary()

# Compare the performance with and without data augmentation
data_augmentation = True

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and real-time data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by dataset std
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in 0 to 180 degrees
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1,  # randomly shift images vertically
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        workers=4)
