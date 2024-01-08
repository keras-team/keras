import keras
from keras import layers, ops
from functools import partial
from keras.api_export import keras_export
from keras.applications import imagenet_utils

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
    },
    "mixer_b32_224": {
        "patch_size": 32,
        "num_blocks": 12,
        "embed_dim": 768,
        "img_size": 224,
    },
    "mixer_s16_224": {
        "patch_size": 16,
        "num_blocks": 8,
        "embed_dim": 512,
        "img_size": 224,
    },
    "mixer_s32_224": {
        "patch_size": 32,
        "num_blocks": 8,
        "embed_dim": 512,
    },
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

When calling the `summary()` method after instantiating a ConvNeXt model,
prefer setting the `expand_nested` argument `summary()` to `True` to better
investigate the instantiated model.

Args:
    num_classes:Optional number of classes to classify images
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
    global_pool: Pooling after patches defaults to `avg`, For now, only avg is supported
Returns:
    A model instance.
"""


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class DropPath(layers.Layer):
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
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""
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
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    norm_layer=None,
    flatten=True,
    bias=True,
    name=None,
):
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
    num_classes=1000,
    img_size=224,
    in_chans=3,
    patch_size=16,
    num_blocks=8,
    embed_dim=512,
    mlp_ratio=(0.5, 4.0),
    block_layer=MixerBlock,
    mlp_layer=Mlp,
    norm_layer=partial(layers.LayerNormalization, epsilon=1e-6),
    act_layer=ops.gelu,
    drop_rate=0.0,
    proj_drop_rate=0.0,
    drop_path_rate=0.0,
    stem_norm=False,
    global_pool="avg",
    data_format="channels_last",
):
    assert (
        keras.config.image_data_format() == "channels_last"
    ), "Mixer only supports `channels_last` data_format"
    img_size = pair(img_size)
    input_shape = (img_size[0], img_size[1], in_chans)
    inputs = layers.Input(input_shape)
    x = PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
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

    if global_pool == "avg":
        x = ops.mean(x, axis=1)
    x = layers.Dropout(drop_rate)(x)
    if num_classes > 0:
        head = layers.Dense(num_classes, name="head")
    else:
        head = layers.Identity()
    out = head(x)
    return keras.Model(inputs=inputs, outputs=out)


def get_weights_pretrained(model, model_name):
    print(WEIGHTS_NAMES.keys())
    assert (
        model_name in WEIGHTS_NAMES.keys()
    ), f"Weights not available for model {model_name}"
    weights_path = keras.utils.get_file(
        origin=f"{BASE_WEIGHTS_PATH}/{WEIGHTS_NAMES[model_name]}",
    )
    model.load_weights(weights_path)
    return True


@keras_export(
    [
        "keras.applications.mlp_mixer.Mixer_L16_224",
        "keras.applications.Mixer_L16_224",
    ]
)
def Mixer_L16_224(pretrained=False, **kwargs):
    model = MlpMixer(**MODEL_CONFIGS["mixer_l16_224"], **kwargs)
    if pretrained:
        get_weights_pretrained(model, "mixer_l16_224")
    return model


@keras_export(
    [
        "keras.applications.mlp_mixer.Mixer_B16_224",
        "keras.applications.Mixer_B16_224",
    ]
)
def Mixer_B16_224(pretrained=False, **kwargs):
    model = MlpMixer(**MODEL_CONFIGS["mixer_b16_224"], **kwargs)
    if pretrained:
        get_weights_pretrained(model, "mixer_b16_224")
    return model


@keras_export(
    [
        "keras.applications.mlp_mixer.Mixer_B32_224",
        "keras.applications.Mixer_B32_224",
    ]
)
def Mixer_B32_224(pretrained=False, **kwargs):
    model = MlpMixer(**MODEL_CONFIGS["mixer_b32_224"], **kwargs)
    if pretrained:
        get_weights_pretrained(model, "mixer_b32_224")
    return model


@keras_export(
    [
        "keras.applications.mlp_mixer.Mixer_S16_224",
        "keras.applications.Mixer_S16_224",
    ]
)
def Mixer_S16_224(pretrained=False, **kwargs):
    model = MlpMixer(**MODEL_CONFIGS["mixer_s16_224"], **kwargs)
    if pretrained:
        get_weights_pretrained(model, "mixer_s16_224")
    return model


@keras_export(
    [
        "keras.applications.mlp_mixer.Mixer_S16_224",
        "keras.applications.Mixer_S16_224",
    ]
)
def Mixer_S32_224(pretrained=False, **kwargs):
    model = MlpMixer(**MODEL_CONFIGS["mixer_s32_224"], **kwargs)
    if pretrained:
        get_weights_pretrained(model, "mixer_s32_224")
    return model


Mixer_L16_224.__doc__ = BASE_DOCSTRING.format(name="Mixer_L16_224")
Mixer_B16_224.__doc__ = BASE_DOCSTRING.format(name="Mixer_B16_224")
Mixer_B32_224.__doc__ = BASE_DOCSTRING.format(name="Mixer_B32_224")
Mixer_S16_224.__doc__ = BASE_DOCSTRING.format(name="Mixer_S16_224")
Mixer_S32_224.__doc__ = BASE_DOCSTRING.format(name="Mixer_S32_224")


@keras_export("keras.applications.mlp_mixer.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__


@keras_export("keras.applications.convnext.preprocess_input")
def preprocess_input(x, data_format=None):
    """Preprocesses input image in 0-1 by dividing it by 255.0" """
    return x / 255.0
