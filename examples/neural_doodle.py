'''Neural doodle with Keras

# Script Usage

## Arguments
```
--nlabels:              # of regions (colors) in mask images
--style-image:          image to learn style from
--style-mask:           semantic labels for style image
--target-mask:          semantic labels for target image (your doodle)
--content-image:        optional image to learn content from
--target-image-prefix:  path prefix for generated target images
```

## Example 1: doodle using a style image, style mask
and target mask.
```
python neural_doodle.py --nlabels 4 --style-image Monet/style.png \
--style-mask Monet/style_mask.png --target-mask Monet/target_mask.png \
--target-image-prefix generated/monet
```

## Example 2: doodle using a style image, style mask,
target mask and an optional content image.
```
python neural_doodle.py --nlabels 4 --style-image Renoir/style.png \
--style-mask Renoir/style_mask.png --target-mask Renoir/target_mask.png \
--content-image Renoir/creek.jpg \
--target-image-prefix generated/renoir
```

# References

- [Dmitry Ulyanov's blog on fast-neural-doodle](http://dmitryulyanov.github.io/feed-forward-neural-doodle/)
- [Torch code for fast-neural-doodle](https://github.com/DmitryUlyanov/fast-neural-doodle)
- [Torch code for online-neural-doodle](https://github.com/DmitryUlyanov/online-neural-doodle)
- [Paper Texture Networks: Feed-forward Synthesis of Textures and Stylized Images](http://arxiv.org/abs/1603.03417)
- [Discussion on parameter tuning](https://github.com/keras-team/keras/issues/3705)

# Resources

Example images can be downloaded from
https://github.com/DmitryUlyanov/fast-neural-doodle/tree/master/data
'''
from __future__ import print_function
import time
import argparse
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imread, imsave

from keras import backend as K
from keras.layers import Input, AveragePooling2D
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19

# Command line arguments
parser = argparse.ArgumentParser(description='Keras neural doodle example')
parser.add_argument('--nlabels', type=int,
                    help='number of semantic labels'
                    ' (regions in differnet colors)'
                    ' in style_mask/target_mask')
parser.add_argument('--style-image', type=str,
                    help='path to image to learn style from')
parser.add_argument('--style-mask', type=str,
                    help='path to semantic mask of style image')
parser.add_argument('--target-mask', type=str,
                    help='path to semantic mask of target image')
parser.add_argument('--content-image', type=str, default=None,
                    help='path to optional content image')
parser.add_argument('--target-image-prefix', type=str,
                    help='path prefix for generated results')
args = parser.parse_args()

style_img_path = args.style_image
style_mask_path = args.style_mask
target_mask_path = args.target_mask
content_img_path = args.content_image
target_img_prefix = args.target_image_prefix
use_content_img = content_img_path is not None

num_labels = args.nlabels
num_colors = 3  # RGB
# determine image sizes based on target_mask
ref_img = imread(target_mask_path)
img_nrows, img_ncols = ref_img.shape[:2]

total_variation_weight = 50.
style_weight = 1.
content_weight = 0.1 if use_content_img else 0

content_feature_layers = ['block5_conv2']
# To get better generation qualities, use more conv layers for style features
style_feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                        'block4_conv1', 'block5_conv1']


# helper functions for reading/processing images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def kmeans(xs, k):
    assert xs.ndim == 2
    try:
        from sklearn.cluster import k_means
        _, labels, _ = k_means(xs.astype('float64'), k)
    except ImportError:
        from scipy.cluster.vq import kmeans2
        _, labels = kmeans2(xs, k, missing='raise')
    return labels


def load_mask_labels():
    '''Load both target and style masks.
    A mask image (nr x nc) with m labels/colors will be loaded
    as a 4D boolean tensor: (1, m, nr, nc) for 'channels_first' or (1, nr, nc, m) for 'channels_last'
    '''
    target_mask_img = load_img(target_mask_path,
                               target_size=(img_nrows, img_ncols))
    target_mask_img = img_to_array(target_mask_img)
    style_mask_img = load_img(style_mask_path,
                              target_size=(img_nrows, img_ncols))
    style_mask_img = img_to_array(style_mask_img)
    if K.image_data_format() == 'channels_first':
        mask_vecs = np.vstack([style_mask_img.reshape((3, -1)).T,
                               target_mask_img.reshape((3, -1)).T])
    else:
        mask_vecs = np.vstack([style_mask_img.reshape((-1, 3)),
                               target_mask_img.reshape((-1, 3))])

    labels = kmeans(mask_vecs, num_labels)
    style_mask_label = labels[:img_nrows *
                              img_ncols].reshape((img_nrows, img_ncols))
    target_mask_label = labels[img_nrows *
                               img_ncols:].reshape((img_nrows, img_ncols))

    stack_axis = 0 if K.image_data_format() == 'channels_first' else -1
    style_mask = np.stack([style_mask_label == r for r in range(num_labels)],
                          axis=stack_axis)
    target_mask = np.stack([target_mask_label == r for r in range(num_labels)],
                           axis=stack_axis)

    return (np.expand_dims(style_mask, axis=0),
            np.expand_dims(target_mask, axis=0))

# Create tensor variables for images
if K.image_data_format() == 'channels_first':
    shape = (1, num_colors, img_nrows, img_ncols)
else:
    shape = (1, img_nrows, img_ncols, num_colors)

style_image = K.variable(preprocess_image(style_img_path))
target_image = K.placeholder(shape=shape)
if use_content_img:
    content_image = K.variable(preprocess_image(content_img_path))
else:
    content_image = K.zeros(shape=shape)

images = K.concatenate([style_image, target_image, content_image], axis=0)

# Create tensor variables for masks
raw_style_mask, raw_target_mask = load_mask_labels()
style_mask = K.variable(raw_style_mask.astype('float32'))
target_mask = K.variable(raw_target_mask.astype('float32'))
masks = K.concatenate([style_mask, target_mask], axis=0)

# index constants for images and tasks variables
STYLE, TARGET, CONTENT = 0, 1, 2

# Build image model, mask model and use layer outputs as features
# image model as VGG19
image_model = vgg19.VGG19(include_top=False, input_tensor=images)

# mask model as a series of pooling
mask_input = Input(tensor=masks, shape=(None, None, None), name='mask_input')
x = mask_input
for layer in image_model.layers[1:]:
    name = 'mask_%s' % layer.name
    if 'conv' in layer.name:
        x = AveragePooling2D((3, 3), padding='same', strides=(
            1, 1), name=name)(x)
    elif 'pool' in layer.name:
        x = AveragePooling2D((2, 2), name=name)(x)
mask_model = Model(mask_input, x)

# Collect features from image_model and task_model
image_features = {}
mask_features = {}
for img_layer, mask_layer in zip(image_model.layers, mask_model.layers):
    if 'conv' in img_layer.name:
        assert 'mask_' + img_layer.name == mask_layer.name
        layer_name = img_layer.name
        img_feat, mask_feat = img_layer.output, mask_layer.output
        image_features[layer_name] = img_feat
        mask_features[layer_name] = mask_feat


# Define loss functions
def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(x)
    gram = K.dot(features, K.transpose(features))
    return gram


def region_style_loss(style_image, target_image, style_mask, target_mask):
    '''Calculate style loss between style_image and target_image,
    for one common region specified by their (boolean) masks
    '''
    assert 3 == K.ndim(style_image) == K.ndim(target_image)
    assert 2 == K.ndim(style_mask) == K.ndim(target_mask)
    if K.image_data_format() == 'channels_first':
        masked_style = style_image * style_mask
        masked_target = target_image * target_mask
        num_channels = K.shape(style_image)[0]
    else:
        masked_style = K.permute_dimensions(
            style_image, (2, 0, 1)) * style_mask
        masked_target = K.permute_dimensions(
            target_image, (2, 0, 1)) * target_mask
        num_channels = K.shape(style_image)[-1]
    num_channels = K.cast(num_channels, dtype='float32')
    s = gram_matrix(masked_style) / K.mean(style_mask) / num_channels
    c = gram_matrix(masked_target) / K.mean(target_mask) / num_channels
    return K.mean(K.square(s - c))


def style_loss(style_image, target_image, style_masks, target_masks):
    '''Calculate style loss between style_image and target_image,
    in all regions.
    '''
    assert 3 == K.ndim(style_image) == K.ndim(target_image)
    assert 3 == K.ndim(style_masks) == K.ndim(target_masks)
    loss = K.variable(0)
    for i in range(num_labels):
        if K.image_data_format() == 'channels_first':
            style_mask = style_masks[i, :, :]
            target_mask = target_masks[i, :, :]
        else:
            style_mask = style_masks[:, :, i]
            target_mask = target_masks[:, :, i]
        loss += region_style_loss(style_image,
                                  target_image, style_mask, target_mask)
    return loss


def content_loss(content_image, target_image):
    return K.sum(K.square(target_image - content_image))


def total_variation_loss(x):
    assert 4 == K.ndim(x)
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] -
                     x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] -
                     x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] -
                     x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] -
                     x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# Overall loss is the weighted sum of content_loss, style_loss and tv_loss
# Each individual loss uses features from image/mask models.
loss = K.variable(0)
for layer in content_feature_layers:
    content_feat = image_features[layer][CONTENT, :, :, :]
    target_feat = image_features[layer][TARGET, :, :, :]
    loss += content_weight * content_loss(content_feat, target_feat)

for layer in style_feature_layers:
    style_feat = image_features[layer][STYLE, :, :, :]
    target_feat = image_features[layer][TARGET, :, :, :]
    style_masks = mask_features[layer][STYLE, :, :, :]
    target_masks = mask_features[layer][TARGET, :, :, :]
    sl = style_loss(style_feat, target_feat, style_masks, target_masks)
    loss += (style_weight / len(style_feature_layers)) * sl

loss += total_variation_weight * total_variation_loss(target_image)
loss_grads = K.gradients(loss, target_image)

# Evaluator class for computing efficiency
outputs = [loss]
if isinstance(loss_grads, (list, tuple)):
    outputs += loss_grads
else:
    outputs.append(loss_grads)

f_outputs = K.function([target_image], outputs)


def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# Generate images by iterative optimization
if K.image_data_format() == 'channels_first':
    x = np.random.uniform(0, 255, (1, 3, img_nrows, img_ncols)) - 128.
else:
    x = np.random.uniform(0, 255, (1, img_nrows, img_ncols, 3)) - 128.

for i in range(50):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_image(x.copy())
    fname = target_img_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
