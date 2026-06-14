import inspect
import math

from keras.src.api_export import keras_export
from keras.src.utils.module_utils import aggdraw
from keras.src.utils.module_utils import pil_image
from keras.src.utils.module_utils import pil_imagedraw
from keras.src.utils.module_utils import pil_imagefont

DEFAULT_COLORS = [
    "#ffd166",
    "#ef476f",
    "#06d6a0",
    "#118ab2",
    "#073b4c",
    "#ffadad",
    "#caffbf",
    "#9bf6ff",
    "#a0c4ff",
    "#bdb2ff",
]


def _fade_color(color, fade_amount):
    """Darken a hex or RGBA color by the given step."""
    if isinstance(color, str) and color.startswith("#"):
        h = color.lstrip("#")
        rgb = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
        color = rgb + (255,)
    elif isinstance(color, tuple):
        if len(color) == 3:
            color = color + (255,)
    else:
        color = (128, 128, 128, 255)
    return tuple(max(0, c - fade_amount) for c in color[:3]) + (color[3],)


class ColorWheel:
    def __init__(self, color_map=None):
        self._cache = {}
        self.color_map = color_map or {}

    def get_color(self, layer):
        if type(layer) in self.color_map:
            c = self.color_map[type(layer)]
            return c.get("fill", c) if isinstance(c, dict) else c

        class_name = layer.__class__.__name__
        for key, color in self.color_map.items():
            if isinstance(key, str) and key in class_name:
                return (
                    color.get("fill", color)
                    if isinstance(color, dict)
                    else color
                )

        if class_name not in self._cache:
            index = len(self._cache) % len(DEFAULT_COLORS)
            self._cache[class_name] = DEFAULT_COLORS[index]
        return self._cache[class_name]


def _extract_shape(layer):
    """Robustly extract the output shape across Keras versions."""
    shape = None
    if hasattr(layer, "output_shape") and layer.output_shape is not None:
        shape = layer.output_shape
    elif hasattr(layer, "output") and hasattr(layer.output, "shape"):
        shape = layer.output.shape
    else:
        return []

    if isinstance(shape, list):
        shape = shape[0]
    if (
        isinstance(shape, tuple)
        and len(shape) > 0
        and isinstance(shape[0], (tuple, list))
    ):
        shape = shape[0]

    return [d for d in shape[1:] if d is not None and isinstance(d, int)]


def _get_layer_dims(shape, scale_z, scale_xy, max_z, max_xy, min_z, min_xy):
    if not shape:
        return min_xy, min_xy, min_z
    if len(shape) == 1:
        z = min(max(shape[0] * scale_z, min_z), max_z)
        return min_xy, min_xy, z
    elif len(shape) == 2:
        x = min(max(shape[0] * scale_xy, min_xy), max_xy)
        y = min(max(shape[1] * scale_xy, min_xy), max_xy)
        z = min(max(shape[1] * scale_z, min_z), max_z)
        return x, y, z
    else:
        x = min(max(shape[0] * scale_xy, min_xy), max_xy)
        y = min(max(shape[1] * scale_xy, min_xy), max_xy)
        z_prod = 1
        for d in shape[2:]:
            z_prod *= d
        z = min(max(z_prod * scale_z, min_z), max_z)
        return x, y, z


def _draw_box(draw, box, shade_step):
    x1, y1, x2, y2, de = box["x1"], box["y1"], box["x2"], box["y2"], box["de"]
    pen = aggdraw.Pen(box["outline"], 1)
    brush_front = aggdraw.Brush(_fade_color(box["fill"], 0))

    if de > 0:
        draw.line([x1 + de, y1 - de, x1 + de, y2 - de], pen)
        draw.line([x1 + de, y2 - de, x1, y2], pen)
        draw.line([x1 + de, y2 - de, x2 + de, y2 - de], pen)

        brush_top = aggdraw.Brush(_fade_color(box["fill"], shade_step))
        brush_side = aggdraw.Brush(_fade_color(box["fill"], shade_step * 2))

        draw.polygon(
            [x1, y1, x1 + de, y1 - de, x2 + de, y1 - de, x2, y1], pen, brush_top
        )

        draw.polygon(
            [x2, y1, x2 + de, y1 - de, x2 + de, y2 - de, x2, y2],
            pen,
            brush_side,
        )

    draw.rectangle([x1, y1, x2, y2], pen, brush_front)


def _draw_connector(draw, b1, b2, outline):
    pen = aggdraw.Pen(outline, 1)
    draw.line(
        [
            b1["x2"] + b1["de"],
            b1["y1"] - b1["de"],
            b2["x1"] + b2["de"],
            b2["y1"] - b2["de"],
        ],
        pen,
    )
    draw.line(
        [
            b1["x2"] + b1["de"],
            b1["y2"] - b1["de"],
            b2["x1"] + b2["de"],
            b2["y2"] - b2["de"],
        ],
        pen,
    )
    draw.line([b1["x2"], b1["y2"], b2["x1"], b2["y2"]], pen)
    draw.line([b1["x2"], b1["y1"], b2["x1"], b2["y1"]], pen)


def _draw_legend(layers, color_wheel, background_fill):
    seen = {}
    for layer in layers:
        class_name = layer.__class__.__name__
        if class_name not in seen:
            seen[class_name] = _fade_color(color_wheel.get_color(layer), 0)

    if not seen:
        return None

    swatch_size = 16
    row_height = 24
    padding = 10

    try:
        font = pil_imagefont.load_default()
        if hasattr(font, "getlength"):
            max_text_width = max(int(font.getlength(n)) for n in seen)
        else:
            max_text_width = max(font.getbbox(n)[2] for n in seen)
    except Exception:
        font = None
        max_text_width = max(len(name) for name in seen) * 8

    legend_w = swatch_size + padding * 3 + max_text_width
    legend_h = padding * 2 + len(seen) * row_height

    img = pil_image.new("RGBA", (legend_w, legend_h), background_fill)
    draw = pil_imagedraw.Draw(img)

    y = padding
    for name, color in seen.items():
        draw.rectangle(
            [padding, y, padding + swatch_size, y + swatch_size],
            fill=color,
            outline=(0, 0, 0, 255),
        )
        draw.text(
            (padding * 2 + swatch_size, y), name, fill=(0, 0, 0, 255), font=font
        )
        y += row_height

    return img


@keras_export("keras.visualization.layered_view")
def layered_view(
    model,
    to_file=None,
    draw_volume=True,
    draw_funnel=True,
    color_map=None,
    text_callable=None,
    legend=False,
    **kwargs,
):
    """Render a Keras model as a layered architecture diagram.

    Note: This visualization assumes a linear sequence of layers.
    For non-linear models (e.g., Functional models with multiple branches
    or skip connections), this visualization will connect layers based
    on their order in `model.layers` rather than the actual graph topology.

    Arguments:
        model: A built Keras model.
        to_file: Optional string or path-like object to save the image.
        draw_volume: Boolean, whether to draw the 3D depth of layers.
        draw_funnel: Boolean, whether to draw connector funnels between layers.
        color_map: Optional dictionary mapping layer classes to colors.
        text_callable: Optional callback function to generate labels for each
            layer. Can take either `(layer) -> str` or `(index, layer) -> str`.
        legend: Boolean, whether to display a color legend.
        **kwargs: Additional styling parameters. Supported arguments include:
            - `min_z`, `min_xy`, `max_z`, `max_xy`: Bounding sizes.
            - `scale_z`, `scale_xy`: Scaling factors.
            - `shade_step`: Color darkening step for 3D faces.
            - `spacing`: Space between layers.
            - `padding`: Canvas padding.
            - `background_fill`: Canvas background color.

    Returns:
        A PIL Image object containing the architecture diagram.

    Example:
    ```python
    model = keras.Sequential([
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(10),
        ]
    )
    keras.visualization.layered_view(model, to_file="model.png")
    ```
    """
    if not model.built:
        raise ValueError(
            f"Model {model.name} has not yet been built. "
            "Build the model first by calling `model.build(input_shape)` "
            "or by passing some data to it."
        )

    min_z = kwargs.get("min_z", 20)
    min_xy = kwargs.get("min_xy", 20)
    max_z = kwargs.get("max_z", 400)
    max_xy = kwargs.get("max_xy", 2000)
    scale_z = kwargs.get("scale_z", 1.5)
    scale_xy = kwargs.get("scale_xy", 4)
    shade_step = kwargs.get("shade_step", 10)
    spacing = kwargs.get("spacing", 10)
    padding = kwargs.get("padding", 10)
    background_fill = kwargs.get("background_fill", "white")

    color_wheel = ColorWheel(color_map)
    boxes = []
    current_z = padding
    x_off = -1

    for layer in model.layers:
        shape = _extract_shape(layer)
        x, y, z = _get_layer_dims(
            shape, scale_z, scale_xy, max_z, max_xy, min_z, min_xy
        )

        de = int(x / 3) if draw_volume else 0
        front_w = int(z)
        front_h = int(y)

        if x_off == -1:
            x_off = de / 2

        box = {
            "layer": layer,
            "x1": current_z - de / 2,
            "y1": de,
            "x2": current_z - de / 2 + front_w,
            "y2": de + front_h,
            "de": de,
            "w": front_w,
            "h": front_h,
            "fill": color_wheel.get_color(layer),
            "outline": (0, 0, 0, 255),
            "total_h": front_h + de,
        }
        boxes.append(box)
        current_z += front_w + spacing

    text_margin = 20 if text_callable else 0

    max_top_extent = max([b["total_h"] / 2 for b in boxes] + [0])
    max_bottom_extent = max([b["total_h"] / 2 for b in boxes] + [0])

    center_y_pos = max_top_extent + padding + text_margin
    img_height = (
        max_top_extent + max_bottom_extent + padding * 2 + text_margin * 2
    )

    min_scene_x = float("inf")
    max_scene_x = float("-inf")
    for box in boxes:
        min_scene_x = min(min_scene_x, box["x1"] + x_off)
        max_scene_x = max(max_scene_x, box["x2"] + x_off + box["de"])

    x_shift = padding - min_scene_x
    img_width = max_scene_x + x_shift + padding

    img = pil_image.new(
        "RGBA",
        (int(math.ceil(img_width)), int(math.ceil(img_height))),
        background_fill,
    )
    draw_img = aggdraw.Draw(img)

    try:
        font = pil_imagefont.load_default()
    except Exception:
        font = None

    last_box = None
    for box in boxes:
        node_top = center_y_pos - box["total_h"] / 2
        box["y1"] = node_top + box["de"]
        box["y2"] = node_top + box["total_h"]
        box["x1"] += x_shift + x_off
        box["x2"] += x_shift + x_off

        if last_box is not None and draw_funnel:
            _draw_connector(draw_img, last_box, box, box["outline"])

        _draw_box(draw_img, box, shade_step)
        last_box = box

    draw_img.flush()
    draw_text = pil_imagedraw.Draw(img)

    if text_callable:
        try:
            expects_index = (
                len(inspect.signature(text_callable).parameters) == 2
            )
        except Exception:
            expects_index = False

        for idx, box in enumerate(boxes):
            res = (
                text_callable(idx, box["layer"])
                if expects_index
                else text_callable(box["layer"])
            )

            if isinstance(res, tuple) and len(res) == 2:
                label, above = res
            else:
                label, above = res, True

            if label:
                try:
                    if hasattr(font, "getlength"):
                        text_w = font.getlength(label)
                    else:
                        text_w = font.getbbox(label)[2]
                except Exception:
                    text_w = len(label) * 8

                if above:
                    t_x = box["x1"] + box["de"] + box["w"] / 2 - text_w / 2
                    t_y = box["y1"] - box["de"] - text_margin
                else:
                    t_x = box["x1"] + box["w"] / 2 - text_w / 2
                    t_y = box["y2"] + text_margin / 2

                draw_text.text(
                    (t_x, t_y), label, fill=(0, 0, 0, 255), font=font
                )

    if legend:
        legend_img = _draw_legend(model.layers, color_wheel, background_fill)
        if legend_img:
            combined = pil_image.new(
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

    if to_file:
        img.save(to_file)

    return img
