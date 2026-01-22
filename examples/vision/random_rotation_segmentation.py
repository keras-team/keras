"""
RandomRotation with segmentation masks (Keras 3).

This example demonstrates how keras.layers.RandomRotation behaves
when applied to images and segmentation masks using structured
dictionary inputs.

We compare:
- fill_mode="constant": introduces border fill values
- fill_mode="crop": removes border artifacts by cropping the valid region
"""

import numpy as np
import matplotlib.pyplot as plt
from keras import layers, ops


def create_sample():
    image = np.zeros((128, 128, 1), dtype="float32")
    image[32:96, 32:96, 0] = 1.0

    mask = np.zeros((128, 128, 1), dtype="int32")
    mask[32:96, 32:96, 0] = 1

    return image, mask


def apply_rotation(layer, image, mask):
    inputs = {
        "images": ops.expand_dims(image, axis=0),
        "segmentation_masks": ops.expand_dims(mask, axis=0),
    }

    outputs = layer(inputs, training=True)

    return (
        ops.squeeze(outputs["images"], axis=0),
        ops.squeeze(outputs["segmentation_masks"], axis=0),
    )


def show(image, mask, title):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    axs[0].imshow(image[..., 0], cmap="gray")
    axs[0].set_title("Image")
    axs[0].axis("off")

    axs[1].imshow(mask[..., 0], cmap="gray")
    axs[1].set_title("Segmentation Mask")
    axs[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    image, mask = create_sample()

    rotation_constant = layers.RandomRotation(
        factor=0.4,
        fill_mode="constant",
        fill_value=0.0,
        seed=42,
    )

    rotation_crop = layers.RandomRotation(
        factor=0.4,
        fill_mode="crop",
        seed=42,
    )

    img_const, mask_const = apply_rotation(rotation_constant, image, mask)
    img_crop, mask_crop = apply_rotation(rotation_crop, image, mask)

    show(image, mask, "Original")
    show(img_const, mask_const, 'RandomRotation(fill_mode="constant")')
    show(img_crop, mask_crop, 'RandomRotation(fill_mode="crop")')


if __name__ == "__main__":
    main()
