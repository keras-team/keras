import keras
from keras import layers, ops
from functools import partial
from keras.api_export import keras_export
from keras.applications import imagenet_utils
from keras.utils import file_utils, get_file
from keras import backend


BASE_WEIGHTS_PATH = "https://huggingface.co/anasrz/kimm/resolve/main"

WEIGHTS_NAMES = {
    "mixer_l16_224": ("jx_mixer_l16_224-92f9adc4.weights.h5"),
    "mixer_b16_224": ("mixer_b16_224_miil-9229a591.weights.h5?download=true"),
}


MODEL_CONFIGS = {
    "mixer_l16_224": {
        "patch_size": 16,
        "num_blocks": 24,
        "embed_dim": 1024,
        "img_size": 224,
    },
    "mixer_b16_224": {
        "patch_size": 16,
        "num_blocks": 12,
        "embed_dim": 768,
        "img_size": 224,
    }
}


BASE_DOCSTRING = """Instantiates the {name} architecture.

References:

- [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)
(NIPS 2021)

For image classification use cases, see
[this page for detailed examples](
https://keras.io/api/applications/#usage-examples-for-image-classification-models).
For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning](
https://keras.io/guides/transfer_learning/).

The models `mixer_l16_224`, `mixer_b16_224` have pre-trained parameters assembled from the
[PyTorch Implementation](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mlp_mixer.py). 
The parameters of ported weights and output logits was verified. You can check usage: verification here:

Note: Each Keras Application expects a specific kind of input preprocessing.
For Mlp Mixer, preprocessing is not included in the model using a `Normalization`
layer.  Mlp Mixer models expect their inputs to be float or uint8 tensors of
pixels with values in the [0-255] range.


Args:
    classes:Optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified. Defaults to 1000 (number of
            ImageNet classes)., Set None or 0 for extracting features
    img_size: Optional input image size int/tuple,
            defaults to 224
    in_chans: Optional number of channels in the inputs, defaults to 3
    patch_size: patch size for for images, defaults to 16,
    num_blocks: number of mixer blocks defaults to 8,
    embed_dim: defaults to 512,
    mlp_ratio: the ratio to which Mlp model inside Mixer is projected to defaults to: (0.5, 4.0),
    block_layer: layer to be used for mixing. defaults to `MixerBlock`,
    mlp_layer: block used to for projecting defaults to `Mlp`,
    norm_layer: Layer used for normalization defaults to `LayerNormalization` with epsilon:1e-6
    act_layer : Activation function to be used `ops.gelu`,
    drop_rate : dropout rate, defaults to 0.,
    proj_drop_rate: dropout rate fro projection defaults to 0.,
    drop_path_rate: dropout rate for DropPath layer defaults to 0.,
    stem_norm: Normalization after extracting patches, defaults to False,
    pooling: Pooling after patches defaults to `avg`, in {'avg', 'max'}
Returns:
    A model instance.
"""


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class DropPath(layers.Layer):
    """Implements DropPath regularization for stochastic depth.

    DropPath is a regularization technique for neural networks that randomly
    drops entire layers or blocks of layers during training. This helps to
    prevent overfitting and encourages the network to learn more robust
    representations.

    Args:
    rate: Probability of dropping a unit.
    seed: Seed for random number generation.
    **kwargs: Additional keyword arguments passed to the Layer constructor.
    """
    def __init__(self, rate=0.5, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self._seed_val = seed
        self.seed = keras.random.SeedGenerator(seed=seed)

    def call(self, x, training=None):
        if self.rate == 0.0 or not training:
            return x
        else:
            batch_size = x.shape[0] or ops.shape(x)[0]
            drop_map_shape = (batch_size,) + (1,) * (len(x.shape) - 1)
            drop_map = ops.cast(
                keras.random.uniform(drop_map_shape, seed=self.seed)
                > self.rate,
                x.dtype,
            )
            x = x / (1.0 - self.rate)
            x = x * drop_map
            return x


def Mlp(
    hidden_features=None,
    out_features=None,
    act_layer=ops.gelu,
    norm_layer=None,
    bias=True,
    drop=0.0,
    use_conv=False,
    name="mlp",
):
    """Creates a multi-layer perceptron (MLP) layer.

    This MLP layer is designed for use in Vision Transformer, MLP-Mixer,
    and related architectures. It offers flexibility in terms of activation
    function, normalization, dropout, and the use of convolutional or dense
    layers for internal computations.

    Args:
    hidden_features: Number of hidden units in the first layer.
                    If None, defaults to the input features.
    out_features: Number of output units. If None, defaults to the input features.
    act_layer: Activation layer to use. Defaults to ops.gelu.
    norm_layer: Normalization layer to apply after each linear layer.
    bias: Whether to include bias terms in the linear layers.
    drop: Dropout rate to apply.
    use_conv: Whether to use convolutional layers instead of dense layers.
    name: Name of the layer.

    Returns:
    A callable that accepts an input tensor and returns the output of the MLP.
    """
    if not name:
        name = ""
    bias = pair(bias)
    drop_probs = pair(drop)
    linear_layer = (
        partial(layers.Conv2D, kernel_size=1) if use_conv else layers.Dense
    )
    norm = (
        norm_layer(name=f"{name}.norm")
        if norm_layer is not None
        else layers.Identity()
    )

    def _apply(x):
        nonlocal out_features, hidden_features
        in_features = ops.shape(x)[-1]
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        x = linear_layer(hidden_features, use_bias=bias[0], name=f"{name}.fc1")(
            x
        )
        x = act_layer(x)
        x = layers.Dropout(drop_probs[0])(x)
        x = norm(x)
        x = linear_layer(out_features, use_bias=bias[1], name=f"{name}.fc2")(x)
        x = layers.Dropout(drop_probs[1])(x)
        return x

    return _apply


def MixerBlock(
    dim,
    seq_len,
    mlp_ratio=(0.5, 4.0),
    mlp_layer=Mlp,
    norm_layer=partial(layers.LayerNormalization, epsilon=1e-6),
    act_layer=ops.gelu,
    drop=0.0,
    drop_path=0.0,
    name="block",
):
    """Creates a Mixer block for MLP-Mixer models.

    This block consists of two MLP layers applied to tokens and channels,
    respectively, with residual connections and optional layer normalization
    and dropout. It's a core component of the MLP-Mixer architecture for
    vision and other tasks.

    Args:
    dim: Dimensionality of the input features.
    seq_len: Sequence length of the input.
    mlp_ratio: Ratio of hidden dimensions in the MLP layers,
                    specified as a tuple of two values.
    mlp_layer: Type of MLP layer to use. Defaults to Mlp.
    norm_layer: Normalization layer to apply after each MLP layer.
    act_layer: Activation layer to use. Defaults to ops.gelu.
    drop: Dropout rate to apply.
    drop_path: Drop path rate for stochastic depth.
    name: Name of the block.

    Returns:
    A callable that accepts an input tensor and returns the output of the block.
    """
    tokens_dim, channels_dim = [int(x * dim) for x in pair(mlp_ratio)]
    drop_path = DropPath(drop_path) if drop_path > 0.0 else layers.Identity()

    def _apply(x):
        x_skip = x
        x = norm_layer(name=f"{name}.norm1")(x)
        x = layers.Permute((2, 1))(x)
        x = mlp_layer(
            tokens_dim,
            act_layer=act_layer,
            drop=drop,
            name=f"{name}.mlp_tokens",
        )(x)
        x = layers.Permute((2, 1))(x)
        x = x_skip + drop_path(x)
        x_skip = x
        x = norm_layer(name=f"{name}.norm2")(x)
        x = mlp_layer(
            channels_dim,
            act_layer=act_layer,
            drop=drop,
            name=f"{name}.mlp_channels",
        )(x)
        x = x_skip + drop_path(x)

        return x

    return _apply


def PatchEmbed(
    patch_size=16,
    embed_dim=768,
    norm_layer=None,
    bias=True,
    name=None,
):
    """Creates a patch embedding layer for ViT architectures.

    This layer divides an input image into patches, embeds each patch into
    an embedding vector, and optionally applies normalization. It's commonly
    used as the first layer in ViT models to prepare image inputs for
    transformer-based processing.

    Args:
    patch_size: Size of the patches to extract from the input image.
                    Should be a single integer or a pair of integers.
    embed_dim: Dimensionality of the embedding vectors.
    norm_layer: Normalization layer to apply after embedding. Defaults to None.
    bias: Whether to include a bias term in the convolutional projection.
    name: Name of the layer.

    Returns:
    A callable that accepts an input tensor and returns the embedded patches.
    """
    patch_size = pair(patch_size)
    norm = norm_layer() if norm_layer else layers.Identity()

    def _apply(x):
        x = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=bias,
            name=f"{name}.proj",
        )(x)
        x = layers.Reshape((-1, embed_dim))(x)
        x = norm(x)
        return x

    return _apply


def MlpMixer(
    classes=1000,
    input_shape=None,
    img_size=224,
    in_chans=3,
    patch_size=16,
    num_blocks=8,
    embed_dim=512,
    mlp_ratio=(0.5, 4.0),
    block_layer=MixerBlock,
    mlp_layer=Mlp,
    include_top=True,
    norm_layer=partial(layers.LayerNormalization, epsilon=1e-6),
    act_layer=ops.gelu,
    drop_rate=0.0,
    proj_drop_rate=0.0,
    drop_path_rate=0.0,
    stem_norm=False,
    pooling="avg",
):
    """Creates an MLP-Mixer model for image classification.

    MLP-Mixer is an architecture that uses multi-layer perceptrons (MLPs)
    instead of attention or convolutions for image processing. It achieves
    competitive results on image classification benchmarks while being
    more efficient and easier to train.

    Args:
    classes: Number of output classes for classification.
    input_shape: Shape of the input image tensor.
    img_size: Size of the input image.
    in_chans: Number of input channels.
    patch_size: Size of the patches to extract from the image.
    num_blocks: Number of Mixer blocks in the model.
    embed_dim: Dimensionality of the embedding vectors.
    mlp_ratio: Ratio of hidden dimensions in the MLP layers.
    block_layer: Type of block layer to use. Defaults to MixerBlock.
    mlp_layer: Type of MLP layer to use. Defaults to Mlp.
    include_top: Whether to include the final classification head.
    norm_layer: Normalization layer to use.
    act_layer: Activation layer to use.
    drop_rate: Dropout rate to apply.
    proj_drop_rate: Dropout rate to apply to projection layers.
    drop_path_rate: Drop path rate for stochastic depth.
    stem_norm: Whether to apply normalization in the stem.
    pooling: Type of pooling to apply before the classification head.
    weights: Optional weights to load into the model.

    Returns:
    A Keras model instance.
    """
    if backend.image_data_format() == "channels_first":
        raise ValueError(
            "Mlp Mixer does not support the `channels_first` image data "
            "format. Switch to `channels_last` by editing your local "
            "config file at ~/.keras/keras.json"
        )
    
    img_size = pair(img_size)
    if not input_shape:
       input_shape = (img_size[0], img_size[1], in_chans)
    inputs = layers.Input(input_shape)
    x = PatchEmbed(
        patch_size=patch_size,
        embed_dim=embed_dim,
        norm_layer=norm_layer if stem_norm else None,
        name="stem",
    )(
        inputs
    )  # stem
    num_patches = ops.shape(x)[1]
    for i in range(num_blocks):
        x = block_layer(
            embed_dim,
            num_patches,
            mlp_ratio,
            mlp_layer=mlp_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            drop=proj_drop_rate,
            drop_path=drop_path_rate,
            name=f"blocks.{i}",
        )(x)
    x = norm_layer(name="norm")(x)
    x = layers.Dropout(drop_rate)(x)
    if pooling == "avg":
        x = ops.mean(x, axis=1)
    elif pooling == "max":
        x = layers.GlobalMaxPooling2D()(x)
    if include_top:
        head = layers.Dense(classes, name="head")
        x = head(x)
    return keras.Model(inputs=inputs, outputs=x)


def get_weights_pretrained(model, name):
    weights_path = get_file(
        origin=f"{BASE_WEIGHTS_PATH}/{WEIGHTS_NAMES[name]}",
    )
    model.load_weights(weights_path)
    return True

def get_mixer(name, weights=None, include_top=False, classes=1000,  **kwargs):
    if not (weights in {"imagenet", None} or file_utils.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded."
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights="imagenet"` with `include_top=True`, '
            "`classes` should be 1000. "
            f"Received classes={classes}"
        )
    MODEL_CONFIGS[name]['classes'] = classes 
    model = MlpMixer(include_top=include_top, **MODEL_CONFIGS[name], **kwargs)
    if weights == 'imagenet':
        get_weights_pretrained(model, name)
        
    elif weights is not None and file_utils.exists(weights):
        model.load_weights(weights)
    
    return model

@keras_export(
    [
        "keras.applications.mlp_mixer.Mixer_L16_224",
        "keras.applications.Mixer_L16_224",
    ]
)
def Mixer_L16_224(**kwargs):
    return get_mixer("mixer_l16_224", **kwargs)


@keras_export(
    [
        "keras.applications.mlp_mixer.Mixer_B16_224",
        "keras.applications.Mixer_B16_224",
    ]
)

def Mixer_B16_224(**kwargs):
    return get_mixer("mixer_b16_224", **kwargs)



Mixer_L16_224.__doc__ = BASE_DOCSTRING.format(name="Mixer_L16_224")
Mixer_B16_224.__doc__ = BASE_DOCSTRING.format(name="Mixer_B16_224")


@keras_export("keras.applications.mlp_mixer.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__


@keras_export("keras.applications.convnext.preprocess_input")
def preprocess_input(x, data_format=None):
    """Preprocesses input image in 0-1 by dividing it by 255.0" """
    return x / 255.0
