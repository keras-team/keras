# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import _image_decoder_data, expect


def generate_checkerboard(width: int, height: int, square_size: int) -> np.ndarray:
    # Create an empty RGB image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the number of squares in each dimension
    num_squares_x = width // square_size
    num_squares_y = height // square_size

    # Generate a random color for each square
    colors = np.random.randint(
        0, 256, size=(num_squares_y, num_squares_x, 3), dtype=np.uint8
    )

    # Iterate over each square
    for i in range(num_squares_y):
        for j in range(num_squares_x):
            # Calculate the position of the current square
            x = j * square_size
            y = i * square_size

            # Get the color for the current square
            color = colors[i, j]

            # Fill the square with the corresponding color
            image[y : y + square_size, x : x + square_size, :] = color

    return image


def _generate_test_data(
    format_: str,
    frozen_data: _image_decoder_data.ImageDecoderData,
    pixel_format: str = "RGB",
    height: int = 32,
    width: int = 32,
    tile_sz: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import PIL.Image
    except ImportError:
        # Since pillow is not installed to generate test data for the ImageDecoder operator
        # directly use the frozen data from _image_decoder_data.py.
        return frozen_data.data, frozen_data.output
    np.random.seed(12345)
    image = generate_checkerboard(height, width, tile_sz)
    image_pil = PIL.Image.fromarray(image)
    with io.BytesIO() as f:
        image_pil.save(f, format=format_)
        data = f.getvalue()
        data_array = np.frombuffer(data, dtype=np.uint8)
    if pixel_format == "BGR":
        output_pil = PIL.Image.open(io.BytesIO(data))
        output = np.array(output_pil)[:, :, ::-1]
    elif pixel_format == "RGB":
        output_pil = PIL.Image.open(io.BytesIO(data))
        output = np.array(output_pil)
    elif pixel_format == "Grayscale":
        output_pil = PIL.Image.open(io.BytesIO(data)).convert("L")
        output = np.array(output_pil)[:, :, np.newaxis]
    else:
        raise ValueError(f"Unsupported pixel format: {pixel_format}")
    return data_array, output


class ImageDecoder(Base):
    @staticmethod
    def export_image_decoder_decode_jpeg_rgb() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="RGB",
        )

        data, output = _generate_test_data(
            "jpeg", _image_decoder_data.image_decoder_decode_jpeg_rgb, "RGB"
        )
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_jpeg_rgb",
        )

    @staticmethod
    def export_image_decoder_decode_jpeg_grayscale() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="Grayscale",
        )

        data, output = _generate_test_data(
            "jpeg", _image_decoder_data.image_decoder_decode_jpeg_grayscale, "Grayscale"
        )
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_jpeg_grayscale",
        )

    @staticmethod
    def export_image_decoder_decode_jpeg_bgr() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="BGR",
        )

        data, output = _generate_test_data(
            "jpeg", _image_decoder_data.image_decoder_decode_jpeg_bgr, "BGR"
        )
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_jpeg_bgr",
        )

    @staticmethod
    def export_image_decoder_decode_jpeg2k_rgb() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="RGB",
        )

        data, output = _generate_test_data(
            "jpeg2000", _image_decoder_data.image_decoder_decode_jpeg2k_rgb, "RGB"
        )
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_jpeg2k_rgb",
        )

    @staticmethod
    def export_image_decoder_decode_bmp_rgb() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="RGB",
        )

        data, output = _generate_test_data(
            "bmp", _image_decoder_data.image_decoder_decode_bmp_rgb, "RGB"
        )
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_bmp_rgb",
        )

    @staticmethod
    def export_image_decoder_decode_png_rgb() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="RGB",
        )

        data, output = _generate_test_data(
            "png", _image_decoder_data.image_decoder_decode_png_rgb, "RGB"
        )
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_png_rgb",
        )

    @staticmethod
    def export_image_decoder_decode_tiff_rgb() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="RGB",
        )

        data, output = _generate_test_data(
            "tiff", _image_decoder_data.image_decoder_decode_tiff_rgb, "RGB"
        )
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_tiff_rgb",
        )

    @staticmethod
    def export_image_decoder_decode_webp_rgb() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="RGB",
        )

        data, output = _generate_test_data(
            "webp", _image_decoder_data.image_decoder_decode_webp_rgb, "RGB"
        )
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_webp_rgb",
        )

    @staticmethod
    def export_image_decoder_decode_pnm_rgb() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="RGB",
        )

        data, output = _generate_test_data(
            "ppm", _image_decoder_data.image_decoder_decode_pnm_rgb, "RGB"
        )
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_pnm_rgb",
        )
