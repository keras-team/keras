#! -*- coding: utf-8 -*-
# Keras implement of Glow
# https://blog.openai.com/glow/

from keras.layers import *
from keras.models import Model
from keras.datasets import cifar10
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
import imageio
import numpy as np
from scipy import misc
import glob
import os

if not os.path.exists('samples'):
    os.mkdir('samples')

imgs = glob.glob('img_align_celeba/*.jpg')

height, width = misc.imread(imgs[0]).shape[:2]
center_height = int((height - width) / 2)

img_size = 64  # for a fast try, please use img_size=32
depth = 10  # orginal paper use depth=32
level = 3  # orginal paper use level for 256*256 CelebA HQ


def imread(f):
    x = misc.imread(f)
    x = x[center_height:center_height + width, :]
    x = misc.imresize(x, (img_size, img_size))
    return x.astype(np.float32) / 256 - 0.5


def data_generator(batch_size=32):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == batch_size:
                X = np.array(X)
                yield X, X.reshape((X.shape[0], -1))
                X = []


class Permute(Layer):
    """New Permute layer. Reverse or shuffle the final axis of inputs
    """

    def __init__(self, mode='reverse', **kwargs):
        super(Permute, self).__init__(**kwargs)
        self.idxs = None
        self.mode = mode

    def build(self, input_shape):
        in_dim = input_shape[-1]
        if self.idxs == None:
            if self.mode == 'reverse':
                self.idxs = self.add_weight(
                    name='idxs',
                    shape=(input_shape[-1], ),
                    dtype='int32',
                    initializer=self.reverse_initializer,
                    trainable=False)
            elif self.mode == 'random':
                self.idxs = self.add_weight(
                    name='idxs',
                    shape=(input_shape[-1], ),
                    dtype='int32',
                    initializer=self.random_initializer,
                    trainable=False)

    def reverse_initializer(self, shape):
        idxs = range(shape[0])
        return idxs[::-1]

    def random_initializer(self, shape):
        idxs = range(shape[0])
        np.random.shuffle(idxs)
        return idxs

    def call(self, inputs):
        num_axis = len(K.int_shape(inputs))
        inputs = K.permute_dimensions(inputs, range(num_axis)[::-1])
        x_outs = K.gather(inputs, self.idxs)
        x_outs = K.permute_dimensions(x_outs, range(num_axis)[::-1])
        return x_outs

    def inverse(self):
        in_dim = K.int_shape(self.idxs)[0]
        reverse_idxs = K.tf.nn.top_k(self.idxs, in_dim)[1][::-1]
        layer = Permute()
        layer.idxs = reverse_idxs
        return layer


class InvDense(Layer):
    """Invertible dense layer of final axis.
    If inputs is image, it equals 1x1 invertible Conv2D.
    """

    def __init__(self, isinverse=False, add_logdet_to_loss=True, **kwargs):
        self.kernel = None
        self.isinverse = isinverse
        self.add_logdet_to_loss = add_logdet_to_loss
        super(InvDense, self).__init__(**kwargs)

    def initializer(self, shape):
        """通过随机正交矩阵进行LU分解初始化
        """
        import scipy as sp
        import scipy.linalg
        random_matrix = sp.random.randn(shape[-1], shape[-1])
        random_orthogonal = sp.linalg.qr(random_matrix)[0]
        p, l, u = sp.linalg.lu(random_orthogonal)
        u_diag_sign = sp.sign(sp.diag(u))
        u_diag_abs_log = sp.log(abs(sp.diag(u)))
        l_mask = 1 - sp.tri(shape[-1]).T
        u_mask = 1 - sp.tri(shape[-1])
        return p, l, u, u_diag_sign, u_diag_abs_log, l_mask, u_mask

    def build(self, input_shape):
        if self.kernel == None:
            (p, l, u, u_diag_sign, u_diag_abs_log, l_mask,
             u_mask) = self.initializer(input_shape)
            self.kernel_p = self.add_weight(
                name='kernel_p',
                shape=p.shape,
                initializer=lambda _: p,
                trainable=False)
            self.kernel_l = self.add_weight(
                name='kernel_l',
                shape=l.shape,
                initializer=lambda _: l,
                trainable=True)
            self.kernel_u = self.add_weight(
                name='kernel_u',
                shape=u.shape,
                initializer=lambda _: u,
                trainable=True)
            self.kernel_u_diag_sign = self.add_weight(
                name='kernel_u_diag_sign',
                shape=u_diag_sign.shape,
                initializer=lambda _: u_diag_sign,
                trainable=False)
            self.kernel_u_diag_abs_log = self.add_weight(
                name='kernel_u_diag_abs_log',
                shape=u_diag_abs_log.shape,
                initializer=lambda _: u_diag_abs_log,
                trainable=True)
            self.kernel_l = self.kernel_l * l_mask + K.eye(input_shape[-1])
            self.kernel_u = self.kernel_u * u_mask + K.tf.diag(
                self.kernel_u_diag_sign * K.exp(self.kernel_u_diag_abs_log))
            self.kernel = K.dot(
                K.dot(self.kernel_p, self.kernel_l), self.kernel_u)

    def call(self, inputs):
        if self.isinverse:
            logdet = K.sum(self.kernel_u_diag_abs_log)
            x_outs = K.dot(inputs, K.tf.matrix_inverse(self.kernel))
        else:
            logdet = -K.sum(self.kernel_u_diag_abs_log)
            x_outs = K.dot(inputs, self.kernel)
        if len(K.int_shape(inputs)) > 2:
            logdet *= K.prod(K.cast(K.shape(inputs)[1:-1], 'float32'))
        if self.add_logdet_to_loss:
            self.add_loss(logdet)
        return x_outs

    def inverse(self):
        layer = InvDense(not self.isinverse)
        layer.kernel = self.kernel
        layer.kernel_u_diag_abs_log = self.kernel_u_diag_abs_log
        return layer


class Split(Layer):
    """split inputs into several parts according pattern
    """

    def __init__(self, pattern=None, **kwargs):
        self.pattern = pattern
        super(Split, self).__init__(**kwargs)

    def call(self, inputs):
        if self.pattern == None:
            in_dim = K.int_shape(inputs)[-1]
            self.pattern = [in_dim // 2, in_dim - in_dim // 2]
        partion = [0] + list(np.cumsum(self.pattern))
        return [inputs[..., i:j] for i, j in zip(partion, partion[1:])]

    def compute_output_shape(self, input_shape):
        return [input_shape[:-1] + (d, ) for d in self.pattern]

    def inverse(self):
        layer = Concat()
        return layer


class Concat(Layer):
    """like Concatenate but add inverse()
    """

    def __init__(self, **kwargs):
        super(Concat, self).__init__(**kwargs)

    def call(self, inputs):
        self.pattern = [K.int_shape(i)[-1] for i in inputs]
        return K.concatenate(inputs, -1)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (sum(self.pattern), )

    def inverse(self):
        layer = Split(self.pattern)
        return layer


class AffineCouple(Layer):
    """Affine Coupling
    """

    def __init__(self, isinverse=False, add_logdet_to_loss=True, **kwargs):
        self.isinverse = isinverse
        self.add_logdet_to_loss = add_logdet_to_loss
        super(AffineCouple, self).__init__(**kwargs)

    def call(self, inputs):
        """if len(inputs) == 3, it equals additive coupling.
        if len(inputs) == 4, it is common affine coupling.
        """
        if len(inputs) == 3:
            x1, x2, shift = inputs
            log_scale = K.constant([0.])
        elif len(inputs) == 4:
            x1, x2, shift, log_scale = inputs
        if self.isinverse:
            logdet = K.sum(K.mean(log_scale, 0))
            x_outs = [x1, K.exp(-log_scale) * (x2 - shift)]
        else:
            logdet = -K.sum(K.mean(log_scale, 0))
            x_outs = [x1, K.exp(log_scale) * x2 + shift]
        if self.add_logdet_to_loss:
            self.add_loss(logdet)
        return x_outs

    def inverse(self):
        layer = AffineCouple(not self.isinverse)
        return layer


class CoupleWrapper:
    """just a wrapper of AffineCouple for simpler use.
    """

    def __init__(self, shift_model, log_scale_model=None, isinverse=False):
        self.shift_model = shift_model
        self.log_scale_model = log_scale_model
        self.layer = AffineCouple(isinverse)

    def __call__(self, inputs, whocare=0):
        x1, x2 = inputs
        shift = self.shift_model(x1)
        if whocare == 0:
            layer = self.layer
        else:
            layer = self.layer.inverse()
        if self.log_scale_model == None:
            return layer([x1, x2, shift])
        else:
            log_scale = self.log_scale_model(x1)
            return layer([x1, x2, shift, log_scale])

    def inverse(self):
        return lambda inputs: self(inputs, 1)


class Actnorm(Layer):
    """Scale and shift
    """

    def __init__(self, isinverse=False, add_logdet_to_loss=True, **kwargs):
        self.log_scale = None
        self.shift = None
        self.isinverse = isinverse
        self.add_logdet_to_loss = add_logdet_to_loss
        super(Actnorm, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = (1, ) * (len(input_shape) - 1) + (input_shape[-1], )
        if self.log_scale == None:
            self.log_scale = self.add_weight(
                name='log_scale',
                shape=kernel_shape,
                initializer='zeros',
                trainable=True)
        if self.shift == None:
            self.shift = self.add_weight(
                name='shift',
                shape=kernel_shape,
                initializer='zeros',
                trainable=True)

    def call(self, inputs):
        if self.isinverse:
            logdet = K.sum(self.log_scale)
            x_outs = K.exp(-self.log_scale) * (inputs - self.shift)
        else:
            logdet = -K.sum(self.log_scale)
            x_outs = K.exp(self.log_scale) * inputs + self.shift
        if len(K.int_shape(inputs)) > 2:
            logdet *= K.prod(K.cast(K.shape(inputs)[1:-1], 'float32'))
        if self.add_logdet_to_loss:
            self.add_loss(logdet)
        return x_outs

    def inverse(self):
        layer = Actnorm(not self.isinverse)
        layer.log_scale = self.log_scale
        layer.shift = self.shift
        return layer


class CondActnorm(Layer):
    """Conditional scale and shift.
    """

    def __init__(self, isinverse=False, add_logdet_to_loss=True, **kwargs):
        self.kernel = None
        self.bias = None
        self.isinverse = isinverse
        self.add_logdet_to_loss = add_logdet_to_loss
        super(CondActnorm, self).__init__(**kwargs)

    def build(self, input_shape):
        in_dim = input_shape[0][-1]
        if self.kernel == None:
            self.kernel = self.add_weight(
                name='kernel',
                shape=(3, 3, in_dim, in_dim * 2),
                initializer='zeros',
                trainable=True)
        if self.bias == None:
            self.bias = self.add_weight(
                name='bias',
                shape=(in_dim * 2, ),
                initializer='zeros',
                trainable=True)

    def call(self, inputs):
        x1, x2 = inputs
        in_dim = K.int_shape(x1)[-1]
        x2_conv2d = K.conv2d(x2, self.kernel, padding='same')
        x2_conv2d = K.bias_add(x2_conv2d, self.bias)
        log_scale, shift = x2_conv2d[..., :in_dim], x2_conv2d[..., in_dim:]
        if self.isinverse:
            logdet = K.sum(K.mean(log_scale, 0))
            x_outs = K.exp(-log_scale) * (x1 - shift)
        else:
            logdet = -K.sum(K.mean(log_scale, 0))
            x_outs = K.exp(log_scale) * x1 + shift
        if self.add_logdet_to_loss:
            self.add_loss(logdet)
        return x_outs

    def inverse(self):
        layer = CondActnorm(not self.isinverse)
        layer.kernel = self.kernel
        layer.bias = self.bias
        return layer


class Reshape(Layer):
    """combination of keras's Reshape and Flatten. And add inverse().
    """

    def __init__(self, shape=None, **kwargs):
        self.shape = shape
        super(Reshape, self).__init__(**kwargs)

    def call(self, inputs):
        self.in_shape = [i or -1 for i in K.int_shape(inputs)]
        if self.shape == None:
            self.shape = [-1, np.prod(self.in_shape[1:])]
        return K.reshape(inputs, self.shape)

    def compute_output_shape(self, input_shape):
        return tuple([i if i != -1 else None for i in self.shape])

    def inverse(self):
        return Reshape(self.in_shape)


class Squeeze(Layer):
    """shape=[h, w, c] ==> shape=[h/n, w/n, n*n*c]
    """

    def __init__(self, factor=2, **kwargs):
        self.factor = factor
        super(Squeeze, self).__init__(**kwargs)

    def call(self, inputs):
        height, width, channel = K.int_shape(inputs)[1:]
        assert height % self.factor == 0 and width % self.factor == 0
        inputs = K.reshape(inputs,
                           (-1, height // self.factor, self.factor,
                            width // self.factor, self.factor, channel))
        inputs = K.permute_dimensions(inputs, (0, 1, 3, 2, 4, 5))
        x_outs = K.reshape(inputs,
                           (-1, height // self.factor, width // self.factor,
                            channel * self.factor**2))
        return x_outs

    def compute_output_shape(self, input_shape):
        height, width, channel = input_shape[1:]
        return (None, height // self.factor, width // self.factor,
                channel * self.factor**2)

    def inverse(self):
        layer = UnSqueeze(self.factor)
        return layer


class UnSqueeze(Layer):
    """shape=h, w, c] ==> shape=[h*n, w*n, c/(n*n)]
    """

    def __init__(self, factor=2, **kwargs):
        self.factor = factor
        super(UnSqueeze, self).__init__(**kwargs)

    def call(self, inputs):
        height, width, channel = K.int_shape(inputs)[1:]
        assert channel % (self.factor**2) == 0
        inputs = K.reshape(
            inputs, (-1, height, width, self.factor, self.factor, channel //
                     (self.factor**2)))
        inputs = K.permute_dimensions(inputs, (0, 1, 3, 2, 4, 5))
        x_outs = K.reshape(
            inputs, (-1, height * self.factor, width * self.factor, channel //
                     (self.factor**2)))
        return x_outs

    def compute_output_shape(self, input_shape):
        height, width, channel = input_shape[1:]
        return (None, height * self.factor, width * self.factor,
                channel // (self.factor**2))

    def inverse(self):
        layer = Squeeze(self.factor)
        return layer


def build_basic_model(in_size, in_channel):
    """basic model for coupling
    """
    _in = Input(shape=(None, None, in_channel))
    _ = _in
    hidden_dim = 512
    _ = Conv2D(hidden_dim, (3, 3), padding='same')(_)
    # _ = Actnorm(add_logdet_to_loss=False)(_)
    _ = Activation('relu')(_)
    _ = Conv2D(hidden_dim, (1, 1), activation='relu', padding='same')(_)
    # _ = Actnorm(add_logdet_to_loss=False)(_)
    _ = Activation('relu')(_)
    _ = Conv2D(
        in_channel, (3, 3), kernel_initializer='zeros', padding='same')(_)
    return Model(_in, _)


squeeze = Squeeze()
inner_layers = []
outer_layers = []
for i in range(5):
    inner_layers.append([])

for i in range(3):
    outer_layers.append([])

x_in = Input(shape=(img_size, img_size, 3))
x = x_in
x_outs = []

# add noise into inputs for stability.
x = Lambda(
    lambda s: K.in_train_phase(s + 1. / 256 * K.random_uniform(K.shape(s)), s)
)(x)

for i in range(level):
    x = squeeze(x)
    for j in range(depth):
        actnorm = Actnorm()
        permute = Permute(mode='random')
        split = Split()
        couple = CoupleWrapper(
            build_basic_model(img_size // 2**(i + 1), 3 * 2**(i + 1)))
        concat = Concat()
        inner_layers[0].append(actnorm)
        inner_layers[1].append(permute)
        inner_layers[2].append(split)
        inner_layers[3].append(couple)
        inner_layers[4].append(concat)
        x = actnorm(x)
        x = permute(x)
        x1, x2 = split(x)
        x1, x2 = couple([x1, x2])
        x = concat([x1, x2])
    if i < level - 1:
        split = Split()
        condactnorm = CondActnorm()
        reshape = Reshape()
        outer_layers[0].append(split)
        outer_layers[1].append(condactnorm)
        outer_layers[2].append(reshape)
        x1, x2 = split(x)
        x_out = condactnorm([x2, x1])
        x_out = reshape(x_out)
        x_outs.append(x_out)
        x = x1
    else:
        for _ in outer_layers:
            _.append(None)

final_actnorm = Actnorm()
final_concat = Concat()
final_reshape = Reshape()

x = final_actnorm(x)
x = final_reshape(x)
x = final_concat(x_outs + [x])

encoder = Model(x_in, x)
encoder.summary()
encoder.compile(
    loss=lambda y_true, y_pred: K.sum(0.5 * y_pred**2, 1),
    optimizer=Adam(1e-4))

# decoder(generator) is inverse of encoder

x_in = Input(shape=K.int_shape(encoder.outputs[0])[1:])
x = x_in

x = final_concat.inverse()(x)
outputs = x[:-1]
x = x[-1]
x = final_reshape.inverse()(x)
x = final_actnorm.inverse()(x)
x1 = x

for i, (split, condactnorm, reshape) in enumerate(zip(*outer_layers)[::-1]):
    if i > 0:
        x1 = x
        x_out = outputs[-i]
        x_out = reshape.inverse()(x_out)
        x2 = condactnorm.inverse()([x_out, x1])
        x = split.inverse()([x1, x2])
    for j, (actnorm, permute, split, couple, concat) in enumerate(
            zip(*inner_layers)[::-1][i * depth:(i + 1) * depth]):
        x1, x2 = concat.inverse()(x)
        x1, x2 = couple.inverse()([x1, x2])
        x = split.inverse()([x1, x2])
        x = permute.inverse()(x)
        x = actnorm.inverse()(x)
    x = squeeze.inverse()(x)

decoder = Model(x_in, x)


def sample(std, path):
    """generate samples per epoch
    """
    n = 9
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            decoder_input_shape = (1, ) + K.int_shape(decoder.inputs[0])[1:]
            z_sample = np.array(np.random.randn(*decoder_input_shape)) * std
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(img_size, img_size, 3)
            figure[i * img_size:(i + 1) * img_size, j * img_size:(
                j + 1) * img_size] = digit
    figure = np.clip((figure + 0.5) * 256, 0, 255)
    imageio.imwrite(path, figure)


class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        path = 'samples/test_%s.png' % epoch
        sample(1, path)
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            encoder.save_weights('./best_encoder.weights')
        elif logs['loss'] > 0 and epoch > 10:
            """In general, loss is less than zero.
            If loss is greater than zero again, it means model has collapsed.
            We need to reload the best model and lower learning rate.
            """
            encoder.load_weights('./best_encoder.weights')
            K.set_value(encoder.optimizer.lr, 1e-5)


evaluator = Evaluate()

# at least 500 epochs
encoder.fit_generator(
    data_generator(), steps_per_epoch=1000, epochs=1000, callbacks=[evaluator])
