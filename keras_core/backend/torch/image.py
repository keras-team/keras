from keras_core.backend.torch.core import convert_to_tensor

RESIZE_METHODS = {}  # populated after torchvision import

UNSUPPORTED_METHODS = (
    "lanczos3",
    "lanczos5",
)


def resize(
    image, size, method="bilinear", antialias=False, data_format="channels_last"
):
    try:
        import torchvision
        from torchvision.transforms import InterpolationMode as im

        RESIZE_METHODS.update(
            {
                "bilinear": im.BILINEAR,
                "nearest": im.NEAREST_EXACT,
                "bicubic": im.BICUBIC,
            }
        )
    except:
        raise ImportError(
            "The torchvision package is necessary to use `resize` with the "
            "torch backend. Please install torchvision."
        )
    if method in UNSUPPORTED_METHODS:
        raise ValueError(
            "Resizing with Lanczos interpolation is "
            "not supported by the PyTorch backend. "
            f"Received: method={method}."
        )
    if method not in RESIZE_METHODS:
        raise ValueError(
            "Invalid value for argument `method`. Expected of one "
            f"{RESIZE_METHODS}. Received: method={method}"
        )
    if not len(size) == 2:
        raise ValueError(
            "Argument `size` must be a tuple of two elements "
            f"(height, width). Received: size={size}"
        )
    size = tuple(size)
    image = convert_to_tensor(image)
    if data_format == "channels_last":
        if image.ndim == 4:
            image = image.permute((0, 3, 1, 2))
        elif image.ndim == 3:
            image = image.permute((2, 0, 1))
        else:
            raise ValueError(
                "Invalid input rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"image.shape={image.shape}"
            )

    resized = torchvision.transforms.functional.resize(
        img=image,
        size=size,
        interpolation=RESIZE_METHODS[method],
        antialias=antialias,
    )
    if data_format == "channels_last":
        if len(image.shape) == 4:
            resized = resized.permute((0, 2, 3, 1))
        elif len(image.shape) == 3:
            resized = resized.permute((1, 2, 0))
    return resized
