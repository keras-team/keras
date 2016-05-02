'''This script demonstrates how to build the Inception v3 architecture
using the Keras functional API.
We are not actually training it here, for lack of appropriate data.

For more information about this architecture, see:

"Rethinking the Inception Architecture for Computer Vision"
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
http://arxiv.org/abs/1512.00567
'''
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Flatten, Dense, Dropout
from keras.layers import Input, merge
from keras.models import Model
from keras import regularizers


# global constants
NB_CLASS = 1000  # number of classes
DIM_ORDERING = 'th'  # 'th' (channels, width, height) or 'tf' (width, height, channels)
WEIGHT_DECAY = 0.  # L2 regularization factor
USE_BN = False  # whether to use batch normalization


def conv2D_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              activation='relu', batch_norm=USE_BN,
              weight_decay=WEIGHT_DECAY, dim_ordering=DIM_ORDERING):
    '''Utility function to apply to a tensor a module conv + BN
    with optional weight decay (L2 weight regularization).
    '''
    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation=activation,
                      border_mode=border_mode,
                      W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer,
                      dim_ordering=dim_ordering)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    return x

# Define image input layer

if DIM_ORDERING == 'th':
    img_input = Input(shape=(3, 299, 299))
    CONCAT_AXIS = 1
elif DIM_ORDERING == 'tf':
    img_input = Input(shape=(299, 299, 3))
    CONCAT_AXIS = 3
else:
    raise Exception('Invalid dim ordering: ' + str(DIM_ORDERING))

# Entry module

x = conv2D_bn(img_input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
x = conv2D_bn(x, 32, 3, 3, border_mode='valid')
x = conv2D_bn(x, 64, 3, 3)
x = MaxPooling2D((3, 3), strides=(2, 2), dim_ordering=DIM_ORDERING)(x)

x = conv2D_bn(x, 80, 1, 1, border_mode='valid')
x = conv2D_bn(x, 192, 3, 3, border_mode='valid')
x = MaxPooling2D((3, 3), strides=(2, 2), dim_ordering=DIM_ORDERING)(x)

# mixed: 35 x 35 x 256

branch1x1 = conv2D_bn(x, 64, 1, 1)

branch5x5 = conv2D_bn(x, 48, 1, 1)
branch5x5 = conv2D_bn(branch5x5, 64, 5, 5)

branch3x3dbl = conv2D_bn(x, 64, 1, 1)
branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)
branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
branch_pool = conv2D_bn(branch_pool, 32, 1, 1)
x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

# mixed_1: 35 x 35 x 288

branch1x1 = conv2D_bn(x, 64, 1, 1)

branch5x5 = conv2D_bn(x, 48, 1, 1)
branch5x5 = conv2D_bn(branch5x5, 64, 5, 5)

branch3x3dbl = conv2D_bn(x, 64, 1, 1)
branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)
branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
branch_pool = conv2D_bn(branch_pool, 64, 1, 1)
x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

# mixed2: 35 x 35 x 288

branch1x1 = conv2D_bn(x, 64, 1, 1)

branch5x5 = conv2D_bn(x, 48, 1, 1)
branch5x5 = conv2D_bn(branch5x5, 64, 5, 5)

branch3x3dbl = conv2D_bn(x, 64, 1, 1)
branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)
branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
branch_pool = conv2D_bn(branch_pool, 64, 1, 1)
x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

# mixed3: 17 x 17 x 768

branch3x3 = conv2D_bn(x, 384, 3, 3, subsample=(2, 2), border_mode='valid')

branch3x3dbl = conv2D_bn(x, 64, 1, 1)
branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)
branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3, subsample=(2, 2), border_mode='valid')

branch_pool = MaxPooling2D((3, 3), strides=(2, 2), dim_ordering=DIM_ORDERING)(x)
x = merge([branch3x3, branch3x3dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

# mixed4: 17 x 17 x 768

branch1x1 = conv2D_bn(x, 192, 1, 1)

branch7x7 = conv2D_bn(x, 128, 1, 1)
branch7x7 = conv2D_bn(branch7x7, 128, 1, 7)
branch7x7 = conv2D_bn(branch7x7, 192, 7, 1)

branch7x7dbl = conv2D_bn(x, 128, 1, 1)
branch7x7dbl = conv2D_bn(branch7x7dbl, 128, 7, 1)
branch7x7dbl = conv2D_bn(branch7x7dbl, 128, 1, 7)
branch7x7dbl = conv2D_bn(branch7x7dbl, 128, 7, 1)
branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

# mixed5: 17 x 17 x 768

branch1x1 = conv2D_bn(x, 192, 1, 1)

branch7x7 = conv2D_bn(x, 160, 1, 1)
branch7x7 = conv2D_bn(branch7x7, 160, 1, 7)
branch7x7 = conv2D_bn(branch7x7, 192, 7, 1)

branch7x7dbl = conv2D_bn(x, 160, 1, 1)
branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 1, 7)
branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

# mixed5: 17 x 17 x 768

branch1x1 = conv2D_bn(x, 192, 1, 1)

branch7x7 = conv2D_bn(x, 160, 1, 1)
branch7x7 = conv2D_bn(branch7x7, 160, 1, 7)
branch7x7 = conv2D_bn(branch7x7, 192, 7, 1)

branch7x7dbl = conv2D_bn(x, 160, 1, 1)
branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 1, 7)
branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

# mixed6: 17 x 17 x 768

branch1x1 = conv2D_bn(x, 192, 1, 1)

branch7x7 = conv2D_bn(x, 160, 1, 1)
branch7x7 = conv2D_bn(branch7x7, 160, 1, 7)
branch7x7 = conv2D_bn(branch7x7, 192, 7, 1)

branch7x7dbl = conv2D_bn(x, 160, 1, 1)
branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)
branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

# mixed7: 17 x 17 x 768

branch1x1 = conv2D_bn(x, 192, 1, 1)

branch7x7 = conv2D_bn(x, 192, 1, 1)
branch7x7 = conv2D_bn(branch7x7, 192, 1, 7)
branch7x7 = conv2D_bn(branch7x7, 192, 7, 1)

branch7x7dbl = conv2D_bn(x, 160, 1, 1)
branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 7, 1)
branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)
branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 7, 1)
branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

# Auxiliary head

aux_logits = AveragePooling2D((5, 5), strides=(3, 3), dim_ordering=DIM_ORDERING)(x)
aux_logits = conv2D_bn(aux_logits, 128, 1, 1)
aux_logits = conv2D_bn(aux_logits, 728, 5, 5, border_mode='valid')
aux_logits = Flatten()(aux_logits)
aux_preds = Dense(NB_CLASS, activation='softmax')(aux_logits)

# mixed8: 8 x 8 x 1280

branch3x3 = conv2D_bn(x, 192, 1, 1)
branch3x3 = conv2D_bn(branch3x3, 320, 3, 3, subsample=(2, 2), border_mode='valid')

branch7x7x3 = conv2D_bn(x, 192, 1, 1)
branch7x7x3 = conv2D_bn(branch7x7x3, 192, 1, 7)
branch7x7x3 = conv2D_bn(branch7x7x3, 192, 7, 1)
branch7x7x3 = conv2D_bn(branch7x7x3, 192, 3, 3, subsample=(2, 2), border_mode='valid')

branch_pool = AveragePooling2D((3, 3), strides=(2, 2), dim_ordering=DIM_ORDERING)(x)
x = merge([branch3x3, branch7x7x3, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

# mixed9: 8 x 8 x 2048

branch1x1 = conv2D_bn(x, 320, 1, 1)

branch3x3 = conv2D_bn(x, 384, 1, 1)
branch3x3_1 = conv2D_bn(branch3x3, 384, 1, 3)
branch3x3_2 = conv2D_bn(branch3x3, 384, 3, 1)
branch3x3 = merge([branch3x3_1, branch3x3_2], mode='concat', concat_axis=CONCAT_AXIS)

branch3x3dbl = conv2D_bn(x, 448, 1, 1)
branch3x3dbl = conv2D_bn(branch3x3dbl, 384, 3, 3)
branch3x3dbl_1 = conv2D_bn(branch3x3dbl, 384, 1, 3)
branch3x3dbl_2 = conv2D_bn(branch3x3dbl, 384, 3, 1)
branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2], mode='concat', concat_axis=CONCAT_AXIS)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

# mixed10: 8 x 8 x 2048

branch1x1 = conv2D_bn(x, 320, 1, 1)

branch3x3 = conv2D_bn(x, 384, 1, 1)
branch3x3_1 = conv2D_bn(branch3x3, 384, 1, 3)
branch3x3_2 = conv2D_bn(branch3x3, 384, 3, 1)
branch3x3 = merge([branch3x3_1, branch3x3_2], mode='concat', concat_axis=CONCAT_AXIS)

branch3x3dbl = conv2D_bn(x, 448, 1, 1)
branch3x3dbl = conv2D_bn(branch3x3dbl, 384, 3, 3)
branch3x3dbl_1 = conv2D_bn(branch3x3dbl, 384, 1, 3)
branch3x3dbl_2 = conv2D_bn(branch3x3dbl, 384, 3, 1)
branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2], mode='concat', concat_axis=CONCAT_AXIS)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

# Final pooling and prediction

x = AveragePooling2D((8, 8), strides=(1, 1), dim_ordering=DIM_ORDERING)(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
preds = Dense(NB_CLASS, activation='softmax')(x)

# Define model

model = Model(input=img_input, output=[preds, aux_preds])
model.compile('rmsprop', 'categorical_crossentropy')

# train via e.g. `model.fit(x_train, [y_train] * 2, batch_size=32, nb_epoch=100)`
# Note that for a large dataset it would be preferable
# to train using `fit_generator` (see Keras docs).
