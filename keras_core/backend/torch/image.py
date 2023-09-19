import torch
import torch.nn.functional as tnn

from keras_core.backend.torch.core import convert_to_tensor

RESIZE_INTERPOLATIONS = {}  # populated after torchvision import

UNSUPPORTED_INTERPOLATIONS = (
    "lanczos3",
    "lanczos5",
)


def resize(
    image,
    size,
    interpolation="bilinear",
    antialias=False,
    data_format="channels_last",
):
    try:
        import torchvision
        from torchvision.transforms import InterpolationMode as im

        RESIZE_INTERPOLATIONS.update(
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
    if interpolation in UNSUPPORTED_INTERPOLATIONS:
        raise ValueError(
            "Resizing with Lanczos interpolation is "
            "not supported by the PyTorch backend. "
            f"Received: interpolation={interpolation}."
        )
    if interpolation not in RESIZE_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{RESIZE_INTERPOLATIONS}. Received: interpolation={interpolation}"
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
        interpolation=RESIZE_INTERPOLATIONS[interpolation],
        antialias=antialias,
    )
    if data_format == "channels_last":
        if len(image.shape) == 4:
            resized = resized.permute((0, 2, 3, 1))
        elif len(image.shape) == 3:
            resized = resized.permute((1, 2, 0))
    return resized


AFFINE_TRANSFORM_INTERPOLATIONS = (
    "nearest",
    "bilinear",
)
AFFINE_TRANSFORM_FILL_MODES = {
    "constant": "zeros",
    "nearest": "border",
    # "wrap",  not supported by torch
    "mirror": "reflection",  # torch's reflection is mirror in other backends
    "reflect": "reflection",  # if fill_mode==reflect, redirect to mirror
}


def _apply_grid_transform(
    img,
    grid,
    interpolation="bilinear",
    fill_mode="zeros",
    fill_value=None,
):
    """
    Modified from https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional/_geometry.py
    """  # noqa: E501

    # We are using context knowledge that grid should have float dtype
    fp = img.dtype == grid.dtype
    float_img = img if fp else img.to(grid.dtype)

    shape = float_img.shape
    # Append a dummy mask for customized fill colors, should be faster than
    # grid_sample() twice
    if fill_value is not None:
        mask = torch.ones(
            (shape[0], 1, shape[2], shape[3]),
            dtype=float_img.dtype,
            device=float_img.device,
        )
        float_img = torch.cat((float_img, mask), dim=1)

    float_img = tnn.grid_sample(
        float_img,
        grid,
        mode=interpolation,
        padding_mode=fill_mode,
        align_corners=True,
    )
    # Fill with required color
    if fill_value is not None:
        float_img, mask = torch.tensor_split(float_img, indices=(-1,), dim=-3)
        mask = mask.expand_as(float_img)
        fill_list = (
            fill_value
            if isinstance(fill_value, (tuple, list))
            else [float(fill_value)]
        )
        fill_img = torch.tensor(
            fill_list, dtype=float_img.dtype, device=float_img.device
        ).view(1, -1, 1, 1)
        if interpolation == "nearest":
            bool_mask = mask < 0.5
            float_img[bool_mask] = fill_img.expand_as(float_img)[bool_mask]
        else:  # 'bilinear'
            # The following is mathematically equivalent to:
            # img * mask + (1.0 - mask) * fill =
            # img * mask - fill * mask + fill =
            # mask * (img - fill) + fill
            float_img = float_img.sub_(fill_img).mul_(mask).add_(fill_img)

    img = float_img.round_().to(img.dtype) if not fp else float_img
    return img


def affine_transform(
    image,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format="channels_last",
):
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{AFFINE_TRANSFORM_INTERPOLATIONS}. Received: "
            f"interpolation={interpolation}"
        )
    if fill_mode not in AFFINE_TRANSFORM_FILL_MODES.keys():
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected of one "
            f"{set(AFFINE_TRANSFORM_FILL_MODES.keys())}. "
            f"Received: fill_mode={fill_mode}"
        )

    image = convert_to_tensor(image)
    transform = convert_to_tensor(transform)

    if image.ndim not in (3, 4):
        raise ValueError(
            "Invalid image rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"image.shape={image.shape}"
        )
    if transform.ndim not in (1, 2):
        raise ValueError(
            "Invalid transform rank: expected rank 1 (single transform) "
            "or rank 2 (batch of transforms). Received input with shape: "
            f"transform.shape={transform.shape}"
        )

    # the default fill_value of tnn.grid_sample is "zeros"
    if fill_mode != "constant" or (fill_mode == "constant" and fill_value == 0):
        fill_value = None

    # unbatched case
    need_squeeze = False
    if image.ndim == 3:
        image = image.unsqueeze(dim=0)
        need_squeeze = True
    if transform.ndim == 1:
        transform = transform.unsqueeze(dim=0)

    if data_format == "channels_last":
        image = image.permute((0, 3, 1, 2))

    batch_size = image.shape[0]
    h, w, c = image.shape[-2], image.shape[-1], image.shape[-3]

    # get indices
    shape = [h, w, c]  # (H, W, C)
    meshgrid = torch.meshgrid(
        *[torch.arange(size) for size in shape], indexing="ij"
    )
    indices = torch.concatenate(
        [torch.unsqueeze(x, dim=-1) for x in meshgrid], dim=-1
    )
    indices = torch.tile(indices, (batch_size, 1, 1, 1, 1))
    indices = indices.to(transform)

    # swap the values
    a0 = transform[:, 0].clone()
    a2 = transform[:, 2].clone()
    b1 = transform[:, 4].clone()
    b2 = transform[:, 5].clone()
    transform[:, 0] = b1
    transform[:, 2] = b2
    transform[:, 4] = a0
    transform[:, 5] = a2

    # deal with transform
    transform = torch.nn.functional.pad(
        transform, pad=[0, 1, 0, 0], mode="constant", value=1
    )
    transform = torch.reshape(transform, (batch_size, 3, 3))
    offset = transform[:, 0:2, 2].clone()
    offset = torch.nn.functional.pad(offset, pad=[0, 1, 0, 0])
    transform[:, 0:2, 2] = 0

    # transform the indices
    coordinates = torch.einsum("Bhwij, Bjk -> Bhwik", indices, transform)
    coordinates = torch.moveaxis(coordinates, source=-1, destination=1)
    coordinates += torch.reshape(a=offset, shape=(*offset.shape, 1, 1, 1))
    coordinates = coordinates[:, 0:2, ..., 0]
    coordinates = coordinates.permute((0, 2, 3, 1))

    # normalize coordinates
    coordinates[:, :, :, 1] = coordinates[:, :, :, 1] / (w - 1) * 2.0 - 1.0
    coordinates[:, :, :, 0] = coordinates[:, :, :, 0] / (h - 1) * 2.0 - 1.0
    grid = torch.stack(
        [coordinates[:, :, :, 1], coordinates[:, :, :, 0]], dim=-1
    )

    affined = _apply_grid_transform(
        image,
        grid,
        interpolation=interpolation,
        # if fill_mode==reflect, redirect to mirror
        fill_mode=AFFINE_TRANSFORM_FILL_MODES[fill_mode],
        fill_value=fill_value,
    )

    if data_format == "channels_last":
        affined = affined.permute((0, 2, 3, 1))
    if need_squeeze:
        affined = affined.squeeze(dim=0)
    return affined
