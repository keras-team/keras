''' Trains a ResNet on the CIFAR10 dataset.
    Greater than 91% test accuracy (0.52 val_loss) after 50 epochs
    48sec per epoch on GTX 1080Ti

    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
'''

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import MaxPooling2D, AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os


batch_size = 32
num_classes = 10
epochs = 100

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# input image dimensions
img_rows, img_cols, channels = x_train.shape[1], x_train.shape[2], x_train.shape[3]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    input_shape = (channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, channels)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

xin = Input(shape=input_shape)

filters = 64
blocks = 4
sub_blocks = 2
x = Conv2D(filters=filters, kernel_size=7, padding='same', strides=2,
                    kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(xin)
x = BatchNormalization()(x)
x = Activation('relu')(x)        

# Orig paper uses max pool after 1st conv. Reaches up 87% acc if use_max_pool = True.
# Cifar10 images are already too small at 32x32 to be maxpooled. So, we skip.
use_max_pool = False
if use_max_pool:
    x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    blocks = 3

for i in range(blocks):
    for j in range(sub_blocks):
        strides = 1
        is_first_layer_but_not_first_block = j==0 and i>0
        if is_first_layer_but_not_first_block: 
            strides = 2
        y = Conv2D(filters=filters, kernel_size=3, padding='same', strides=strides,
                    kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(filters=filters, kernel_size=3, padding='same',
                    kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(y)
        y = BatchNormalization()(y)
        if is_first_layer_but_not_first_block: 
            x = Conv2D(filters=filters, kernel_size=1, padding='same', strides=2,
                    kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(x)
        x = keras.layers.add([x, y])
        x = Activation('relu')(x)

    filters = 2*filters

x = AveragePooling2D()(x)
y = Flatten()(x)
yout = Dense(num_classes, activation='softmax', kernel_initializer="he_normal")(y)
model = Model(inputs=[xin], outputs=[yout])
model.compile(loss='categorical_crossentropy', optimizer=Adam(),
              metrics=['accuracy'])
model.summary()

# Save model and weights
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = "cifar10_resnet_model.hdf5" 
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
callbacks = [checkpoint, lr_reducer]

data_augmentation = True

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)


score = model.evaluate(x_test, y_test, verbose=1)
print("")
print('Test loss:', score[0])
print('Test accuracy:', score[1])

score = model.evaluate_generator(datagen.flow(x_test, y_test,
                                      batch_size=batch_size,
                                      shuffle=False),
                                      steps=x_test.shape[0] // batch_size,
                                      workers=4)
print('Data gen test loss:', score[0])
print('Data gen test accuracy:', score[1])
