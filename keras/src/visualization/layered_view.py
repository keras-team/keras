"""Layered architecture visualization for Keras models.

Renders each layer as a 3D block stack, giving an "at a glance"
understanding of model architecture. Uses Pillow for rendering
with no external dependencies beyond PIL.

Example:

```python
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3),
    keras.layers.Flatten(),
    keras.layers.Dense(10),
])
img = keras.visualization.layered_view(model)
img.show()
```
"""

import math

from keras.src.api_export import keras_export

try:
    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


# Default color map for common layer types.
# Maps layer class name substrings to RGBA colors.
DEFAULT_COLOR_MAP = {
    "InputLayer": (255, 255, 255, 255),
    "Conv": (0, 120, 215, 255),
    "SeparableConv": (0, 90, 180, 255),
    "DepthwiseConv": (0, 90, 180, 255),
    "Dense": (220, 50, 50, 255),
    "Pool": (255, 165, 0, 255),
    "BatchNorm": (130, 130, 130, 255),
    "LayerNorm": (130, 130, 130, 255),
    "GroupNorm": (130, 130, 130, 255),
    "Dropout": (180, 180, 180, 255),
    "Flatten": (100, 100, 100, 255),
    "Reshape": (100, 100, 100, 255),
    "Concatenate": (80, 180, 80, 255),
    "Add": (80, 180, 80, 255),
    "Multiply": (80, 180, 80, 255),
    "Attention": (160, 32, 240, 255),
    "Embedding": (255, 215, 0, 255),
    "LSTM": (0, 180, 180, 255),
    "GRU": (0, 180, 180, 255),
    "RNN": (0, 180, 180, 255),
    "Activation": (200, 200, 200, 255),
}
_FALLBACK_COLOR = (80, 80, 80, 255)


def _check_pillow():
    if Image is None:
        raise ImportError(
            "You must install Pillow (`pip install Pillow`) "
            "for `layered_view` to work."
        )


def _fade_color(color, step):
    """Darken an RGBA color by the given step."""
    return tuple(max(0, c - step) for c in color[:3]) + (color[3],)


def _get_layer_color(layer, color_map):
    """Get the color for a layer based on its class name."""
    class_name = layer.__class__.__name__
    if color_map:
        for key, color in color_map.items():
            if key in class_name:
                return color
    for key, color in DEFAULT_COLOR_MAP.items():
        if key in class_name:
            return color
    return _FALLBACK_COLOR


def _get_layer_dims(layer, min_z, max_z, min_xy, max_xy, scale_z, scale_xy):
    """Calculate box dimensions from a layer's output shape.

    Returns:
        Tuple of (width, height, depth) in pixels.
    """
    try:
        output_shape = layer.output.shape
    except (AttributeError, ValueError):
        return min_xy, min_xy, min_z

    # output_shape is typically (batch, ..., features)
    # Strip the batch dimension first, then handle None dimensions.
    shape = output_shape[1:]

    if not shape:
        return min_xy, min_xy, min_z

    if len(shape) == 1:
        # Dense, Flatten, etc. -> (units,)
        units = shape[0]
        if units is None:
            return min_xy, min_xy, min_z
        xy = max(min_xy, min(max_xy, int(math.sqrt(units) * scale_xy)))
        return xy, xy, min_z

    if len(shape) == 2:
        # Conv1D, LSTM, etc. -> (steps, features)
        spatial, features = shape
        w = (
            max(min_xy, min(max_xy, int(spatial * scale_xy)))
            if spatial is not None
            else min_xy
        )
        h = w
        d = (
            max(min_z, min(max_z, int(features * scale_z)))
            if features is not None
            else min_z
        )
        return w, h, d

    # Conv2D and higher -> (H, W, C) or (D, H, W, C)
    h_val, w_val = shape[0], shape[1]
    features = shape[-1]
    w = (
        max(min_xy, min(max_xy, int(w_val * scale_xy)))
        if w_val is not None
        else min_xy
    )
    h = (
        max(min_xy, min(max_xy, int(h_val * scale_xy)))
        if h_val is not None
        else min_xy
    )
    d = (
        max(min_z, min(max_z, int(features * scale_z)))
        if features is not None
        else min_z
    )
    return w, h, d


def _draw_box_3d(draw, x, y, w, h, d, color, shade_step):
    """Draw a 3D box (front face + top face + right face)."""
    offset = int(d * 0.4)

    # Right face (darkest)
    right_color = _fade_color(color, shade_step * 2)
    right_pts = [
        (x + w, y),
        (x + w + offset, y - offset),
        (x + w + offset, y - offset + h),
        (x + w, y + h),
    ]
    draw.polygon(right_pts, fill=right_color, outline=(0, 0, 0, 255))

    # Top face (medium shade)
    top_color = _fade_color(color, shade_step)
    top_pts = [
        (x, y),
        (x + offset, y - offset),
        (x + w + offset, y - offset),
        (x + w, y),
    ]
    draw.polygon(top_pts, fill=top_color, outline=(0, 0, 0, 255))

    # Front face
    draw.rectangle([x, y, x + w, y + h], fill=color, outline=(0, 0, 0, 255))


def _draw_box_2d(draw, x, y, w, h, color):
    """Draw a 2D box (rectangle only)."""
    draw.rectangle([x, y, x + w, y + h], fill=color, outline=(0, 0, 0, 255))


def _draw_funnel_connector(draw, x1, y1, h1, x2, y2, h2, color):
    """Draw trapezoidal connector between two layers."""
    pts = [
        (x1, y1),
        (x1, y1 + h1),
        (x2, y2 + h2),
        (x2, y2),
    ]
    draw.polygon(pts, fill=color + (40,), outline=color + (100,))


def _draw_legend(layers, color_map, background_fill):
    """Create a legend image for the layer types."""
    _check_pillow()
    seen = {}
    for layer in layers:
        class_name = layer.__class__.__name__
        if class_name not in seen:
            seen[class_name] = _get_layer_color(layer, color_map)

    if not seen:
        return None

    swatch_size = 16
    row_height = 24
    padding = 10
    try:
        font = ImageFont.load_default()
        max_text_width = max(int(font.getlength(n)) for n in seen)
    except (AttributeError, OSError):
        font = None
        max_text_width = max(len(name) for name in seen) * 8

    legend_w = swatch_size + padding * 3 + max_text_width
    legend_h = padding * 2 + len(seen) * row_height

    img = Image.new("RGBA", (legend_w, legend_h), background_fill)
    draw = ImageDraw.Draw(img)

    y = padding
    for name, color in seen.items():
        draw.rectangle(
            [padding, y, padding + swatch_size, y + swatch_size],
            fill=color,
            outline=(0, 0, 0, 255),
        )
        draw.text(
            (padding * 2 + swatch_size, y),
            name,
            fill=(0, 0, 0, 255),
            font=font,
        )
        y += row_height

    return img


@keras_export("keras.visualization.layered_view")
def layered_view(
    model,
    to_file=None,
    min_z=20,
    min_xy=20,
    max_z=400,
    max_xy=2000,
    scale_z=1.5,
    scale_xy=4,
    draw_volume=True,
    draw_funnel=True,
    shade_step=10,
    color_map=None,
    spacing=50,
    padding=20,
    background_fill="white",
    text_callable=None,
    legend=False,
):
    """Render a Keras model as a layered architecture diagram.

    Each layer is drawn as a 3D block whose dimensions reflect
    the layer's output shape. The result is a Pillow ``Image``
    that can be displayed inline in Jupyter or saved to a file.

    Args:
        model: A built Keras model instance.
        to_file: Optional file path to save the image
            (e.g. ``"model.png"``). Format is inferred from
            the extension.
        min_z: Minimum depth (pixels) for a layer block.
        min_xy: Minimum width/height (pixels) for a layer block.
        max_z: Maximum depth (pixels) for a layer block.
        max_xy: Maximum width/height (pixels) for a layer block.
        scale_z: Multiplier for depth dimension scaling.
        scale_xy: Multiplier for spatial dimension scaling.
        draw_volume: If ``True``, draw 3D blocks. If ``False``,
            draw flat 2D rectangles.
        draw_funnel: If ``True``, draw trapezoidal connectors
            between consecutive layers.
        shade_step: Lightness step for 3D face shading.
        color_map: Dict mapping layer class name substrings to
            RGBA tuples, e.g. ``{"Conv": (0, 120, 215, 255)}``.
            Falls back to built-in defaults for unmatched layers.
        spacing: Horizontal spacing (pixels) between layers.
        padding: Padding (pixels) around the image.
        background_fill: Background color (color name or tuple).
        text_callable: Optional callable ``f(layer) -> str`` to
            generate text labels drawn above each layer block.
        legend: If ``True``, append a color legend to the image.

    Returns:
        A ``PIL.Image.Image`` instance. Also saved to ``to_file``
        if provided.

    Raises:
        ImportError: If Pillow is not installed.
        ValueError: If the model has not been built.

    Example:

    ```python
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(10),
    ])
    img = keras.visualization.layered_view(model)
    img.show()
    ```
    """
    _check_pillow()

    if not model.built:
        raise ValueError(
            "This model has not yet been built. "
            "Build the model first by calling `build()` or by calling "
            "the model on a batch of data."
        )

    layers = model.layers

    # Calculate dimensions for each layer.
    layer_info = []
    for layer in layers:
        w, h, d = _get_layer_dims(
            layer, min_z, max_z, min_xy, max_xy, scale_z, scale_xy
        )
        color = _get_layer_color(layer, color_map)
        layer_info.append((layer, w, h, d, color))

    # Calculate total image size.
    max_depth_offset = 0
    if draw_volume:
        max_depth_offset = max(int(d * 0.4) for _, _, _, d, _ in layer_info)

    total_width = padding * 2
    max_height = 0
    for _, w, h, d, _ in layer_info:
        total_width += w + spacing
        block_h = h + (int(d * 0.4) if draw_volume else 0)
        max_height = max(max_height, block_h)
    total_width -= spacing  # Remove trailing spacing
    total_height = max_height + padding * 2 + max_depth_offset

    # Add space for text labels.
    text_margin = 20 if text_callable else 0
    total_height += text_margin

    img = Image.new(
        "RGBA",
        (total_width, total_height),
        background_fill,
    )
    draw = ImageDraw.Draw(img)

    # Draw layers left to right, vertically centered.
    center_y = total_height // 2 + max_depth_offset // 2
    x_cursor = padding

    prev_right_x = None
    prev_y = None
    prev_h = None

    for layer, w, h, d, color in layer_info:
        # Center vertically.
        y = center_y - h // 2

        # Draw text label above.
        if text_callable:
            label = text_callable(layer)
            if label:
                draw.text(
                    (x_cursor + w // 2, y - text_margin),
                    label,
                    fill=(0, 0, 0, 255),
                    anchor="mt",
                )

        # Draw funnel connector to previous layer.
        if draw_funnel and prev_right_x is not None:
            _draw_funnel_connector(
                draw,
                prev_right_x,
                prev_y,
                prev_h,
                x_cursor,
                y,
                h,
                (200, 200, 200),
            )

        # Draw the layer block.
        if draw_volume:
            _draw_box_3d(draw, x_cursor, y, w, h, d, color, shade_step)
        else:
            _draw_box_2d(draw, x_cursor, y, w, h, color)

        prev_right_x = x_cursor + w
        prev_y = y
        prev_h = h
        x_cursor += w + spacing

    # Append legend if requested.
    if legend:
        legend_img = _draw_legend(layers, color_map, background_fill)
        if legend_img:
            combined = Image.new(
                "RGBA",
                (
                    max(img.width, legend_img.width),
                    img.height + legend_img.height,
                ),
                background_fill,
            )
            combined.paste(img, (0, 0))
            combined.paste(legend_img, (0, img.height))
            img = combined

    # Save to file if requested.
    if to_file:
        img.save(to_file)

    return img
