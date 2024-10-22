import math

from keras.src.utils import backend_utils


class BoundingBox:
    def __init__(self):
        self.backend = backend_utils.DynamicBackend()

    def convert_format(
        self,
        boxes,
        source: str,
        target: str,
        height=None,
        width=None,
        dtype="float32",
    ):
        """Converts `boxes` from one format to another.

        Supported formats are:

        - `"xyxy"`, also known as `corners` format. In this format the first
            four axes represent `[left, top, right, bottom]` in that order.
        - `"rel_xyxy"`. In this format, the axes are the same as `"xyxy"` but
            the x coordinates are normalized using the image width, and the y
            axes the image height. All values in `rel_xyxy` are in the range
            `(0, 1)`.
        - `"xywh"`. In this format the first four axes represent
            `[left, top, width, height]`.
        - `"rel_xywh". In this format the first four axes represent
            [left, top, width, height], just like `"xywh"`. Unlike `"xywh"`,
            the values are in the range (0, 1) instead of absolute pixel values.
        - `"center_xyWH"`. In this format the first two coordinates represent
            the x and y coordinates of the center of the bounding box, while the
            last two represent the width and height of the bounding box.
        - `"center_yxHW"`. In this format the first two coordinates represent
            the y and x coordinates of the center of the bounding box, while the
            last two represent the height and width of the bounding box.
        - `"yxyx"`. In this format the first four axes represent
            [top, left, bottom, right] in that order.
        - `"rel_yxyx"`. In this format, the axes are the same as `"yxyx"` but
            the x coordinates are normalized using the image width, and the y
            axes the image height. All values in `rel_yxyx` are in the range
            (0, 1).
        Formats are case insensitive. It is recommended that you capitalize
        width and height to maximize the visual difference between `"xyWH"`
        and `"xyxy"`.

        Relative formats, abbreviated `rel`, make use of the shapes of the
        `images` passed. In these formats, the coordinates, widths, and heights
        are all specified as percentages of the host image.

        Example:

        ```python
        boxes = {
            "boxes": [TODO],
            "labels": [TODO],
        }
        boxes_in_xywh = keras.utils.bounding_boxes.convert_format(
            boxes,
            source='xyxy',
            target='xyWH'
        )
        ```

        Args:
            boxes: tensor representing bounding boxes in the format specified in
                the `source` parameter. `boxes` can optionally have extra
                dimensions stacked on the final axis to store metadata. boxes
                should be a 3D tensor, with the shape
                `[batch_size, num_boxes, 4]`. Alternatively, boxes can be a
                dictionary with key 'boxes' containing a tensor matching the
                aforementioned spec.
            source:One of `"xyxy"`, `"yxyx"`, `"xywh"`, `"center_xywh"`,
                `"center_yxhw"`, `"rel_xyxy"`, "rel_yxyx", "rel_xywh",
                "rel_center_xywh". Used to specify the original format of the
                `boxes` parameter.
            target:One of `"xyxy"`, `"yxyx"`, `"xywh"`, `"center_xywh"`,
                `"center_yxhw"`, `"rel_xyxy"`, "rel_yxyx", "rel_xywh",
                "rel_center_xywh". Used to specify the destination format of
                the `boxes` parameter.
            images: (Optional) a batch of images aligned with `boxes` on the
                first axis. Should be at least 3 dimensions, with the first 3
                dimensions representing: `[batch_size, height, width]`. Used in
                some converters to compute relative pixel values of the bounding
                box dimensions. Required when transforming from a rel format to
                a non-rel format.
            dtype: the data type to use when transforming the boxes, defaults to
                `"float32"`.
        """
        if isinstance(boxes, dict):
            boxes["boxes"] = self.convert_format(
                boxes["boxes"],
                source=source,
                target=target,
                height=height,
                width=width,
                dtype=dtype,
            )
            return boxes

        to_xyxy_converters = {
            "xyxy": self._xyxy_to_xyxy,
            "yxyx": self._yxyx_to_xyxy,
            "xywh": self._xywh_to_xyxy,
            "center_xywh": self._center_xywh_to_xyxy,
            "center_yxhw": self._center_yxhw_to_xyxy,
            "rel_xyxy": self._rel_xyxy_to_xyxy,
            "rel_yxyx": self._rel_yxyx_to_xyxy,
            "rel_xywh": self._rel_xywh_to_xyxy,
            "rel_center_xywh": self._rel_center_xywh_to_xyxy,
        }
        from_xyxy_converters = {
            "xyxy": self._xyxy_to_xyxy,
            "yxyx": self._xyxy_to_yxyx,
            "xywh": self._xyxy_to_xywh,
            "center_xywh": self._xyxy_to_center_xywh,
            "center_yxhw": self._xyxy_to_center_yxhw,
            "rel_xyxy": self._xyxy_to_rel_xyxy,
            "rel_yxyx": self._xyxy_to_rel_yxyx,
            "rel_xywh": self._xyxy_to_rel_xywh,
            "rel_center_xywh": self._xyxy_to_rel_center_xywh,
        }

        ops = self.backend
        boxes_shape = ops.shape(boxes)
        if boxes_shape[-1] != 4:
            raise ValueError(
                "`boxes` must be a tensor with the last dimension of 4. "
                f"Received: boxes.shape={boxes_shape}"
            )
        source = source.lower()
        target = target.lower()
        if source not in to_xyxy_converters.keys():
            raise ValueError(
                f"Available source: {list(to_xyxy_converters.keys())}. "
                f"Received: source={source}"
            )
        if target not in from_xyxy_converters.keys():
            raise ValueError(
                f"Available target: {list(from_xyxy_converters.keys())}. "
                f"Received: target={target}"
            )
        boxes = ops.cast(boxes, dtype)
        if source == target:
            return boxes
        if height is not None:
            height = ops.cast(height, dtype)
        if width is not None:
            width = ops.cast(width, dtype)

        if source.startswith("rel_") and target.startswith("rel_"):
            source = source.replace("rel_", "", 1)
            target = target.replace("rel_", "", 1)
        to_xyxy_converter = to_xyxy_converters[source]
        from_xyxy_converter = from_xyxy_converters[target]
        in_xyxy_boxes = to_xyxy_converter(boxes, height, width)
        return from_xyxy_converter(in_xyxy_boxes, height, width)

    def clip_to_image_size(
        self, bounding_boxes, height=None, width=None, format="xyxy"
    ):
        if format not in ("xyxy", "rel_xyxy"):
            raise NotImplementedError
        if format == "xyxy" and (height is None or width is None):
            raise ValueError(
                "`height` and `width` must be set if `format='xyxy'`."
            )

        ops = self.backend
        boxes, labels = bounding_boxes["boxes"], bounding_boxes["labels"]

        if format == "xyxy":
            x1, y1, x2, y2 = ops.numpy.split(boxes, 4, axis=-1)
            x1 = ops.numpy.clip(x1, 0, width)
            y1 = ops.numpy.clip(y1, 0, height)
            x2 = ops.numpy.clip(x2, 0, width)
            y2 = ops.numpy.clip(y2, 0, height)
            boxes = ops.numpy.concatenate([x1, y1, x2, y2], axis=-1)

            areas = self._compute_area(boxes)
            areas = ops.numpy.squeeze(areas, axis=-1)
            labels = ops.numpy.where(areas > 0, labels, -1)
        elif format == "rel_xyxy":
            x1, y1, x2, y2 = ops.numpy.split(boxes, 4, axis=-1)
            x1 = ops.numpy.clip(x1, 0.0, 1.0)
            y1 = ops.numpy.clip(y1, 0.0, 1.0)
            x2 = ops.numpy.clip(x2, 0.0, 1.0)
            y2 = ops.numpy.clip(y2, 0.0, 1.0)
            boxes = ops.numpy.concatenate([x1, y1, x2, y2], axis=-1)

            areas = self._compute_area(boxes)
            areas = ops.numpy.squeeze(areas, axis=-1)
            labels = ops.numpy.where(areas > 0, labels, -1)

        result = bounding_boxes.copy()
        result["boxes"] = boxes
        result["labels"] = labels
        return result

    def affine(
        self,
        boxes,
        angle,
        translate_x,
        translate_y,
        scale,
        shear_x,
        shear_y,
        height,
        width,
        center_x=None,
        center_y=None,
    ):
        ops = self.backend

        boxes_shape = ops.shape(boxes)
        batch_size = boxes_shape[0]
        n_boxes = boxes_shape[1]
        if center_x is None:
            center_x = 0.5
        if center_y is None:
            center_y = 0.5

        matrix = self._compute_inverse_affine_matrix(
            center_x,
            center_y,
            angle,
            translate_x,
            translate_y,
            scale,
            shear_x,
            shear_y,
            height,
            width,
        )
        transposed_matrix = ops.numpy.transpose(matrix[:, :2, :], [0, 2, 1])
        points = boxes  # [B, N, 4]
        points = ops.numpy.stack(
            [
                points[..., 0],
                points[..., 1],
                points[..., 2],
                points[..., 1],
                points[..., 2],
                points[..., 3],
                points[..., 0],
                points[..., 3],
            ],
            axis=-1,
        )
        points = ops.numpy.reshape(points, [batch_size, n_boxes, 4, 2])
        points = ops.numpy.concatenate(
            [
                points,
                ops.numpy.ones([batch_size, n_boxes, 4, 1], points.dtype),
            ],
            axis=-1,
        )
        transformed_points = ops.numpy.einsum(
            "bnxy,byz->bnxz", points, transposed_matrix
        )
        boxes_min = ops.numpy.amin(transformed_points, axis=2)
        boxes_max = ops.numpy.amax(transformed_points, axis=2)
        outputs = ops.numpy.concatenate([boxes_min, boxes_max], axis=-1)
        return outputs

    def crop(self, boxes, top, left, height, width):
        ops = self.backend

        x1, y1, x2, y2 = ops.numpy.split(boxes, 4, axis=-1)
        x1 = x1 - left
        y1 = y1 - top
        x2 = x2 - left
        y2 = y2 - top
        x1 = ops.numpy.clip(x1, 0, width)
        y1 = ops.numpy.clip(y1, 0, height)
        x2 = ops.numpy.clip(x2, 0, width)
        y2 = ops.numpy.clip(y2, 0, height)
        outputs = ops.numpy.concatenate([x1, y1, x2, y2], axis=-1)
        return outputs

    def pad(self, boxes, top, left):
        ops = self.backend

        x1, y1, x2, y2 = ops.numpy.split(boxes, 4, axis=-1)
        x1 = x1 + left
        y1 = y1 + top
        x2 = x2 + left
        y2 = y2 + top
        outputs = ops.numpy.concatenate([x1, y1, x2, y2], axis=-1)
        return outputs

    # Converters

    def _xyxy_to_xyxy(self, boxes, height=None, width=None):
        return boxes

    def _yxyx_to_xyxy(self, boxes, height=None, width=None):
        y1, x1, y2, x2 = self.backend.numpy.split(boxes, 4, axis=-1)
        return self.backend.numpy.concatenate([x1, y1, x2, y2], axis=-1)

    def _xywh_to_xyxy(self, boxes, height=None, width=None):
        x1, y1, w, h = self.backend.numpy.split(boxes, 4, axis=-1)
        x2 = x1 + w
        y2 = y1 + h
        return self.backend.numpy.concatenate([x1, y1, x2, y2], axis=-1)

    def _center_xywh_to_xyxy(self, boxes, height=None, width=None):
        ops = self.backend
        cx, cy, w, h = ops.numpy.split(boxes, 4, axis=-1)
        half_w = w / 2.0
        half_h = h / 2.0
        x1 = cx - half_w
        y1 = cy - half_h
        x2 = cx + half_w
        y2 = cy + half_h
        return self.backend.numpy.concatenate([x1, y1, x2, y2], axis=-1)

    def _center_yxhw_to_xyxy(self, boxes, height=None, width=None):
        ops = self.backend
        cy, cx, h, w = ops.numpy.split(boxes, 4, axis=-1)
        half_w = w / 2.0
        half_h = h / 2.0
        x1 = cx - half_w
        y1 = cy - half_h
        x2 = cx + half_w
        y2 = cy + half_h
        return self.backend.numpy.concatenate([x1, y1, x2, y2], axis=-1)

    def _rel_xyxy_to_xyxy(self, boxes, height=None, width=None):
        ops = self.backend
        rel_x1, rel_y1, rel_x2, rel_y2 = ops.numpy.split(boxes, 4, axis=-1)
        x1 = rel_x1 * width
        y1 = rel_y1 * height
        x2 = rel_x2 * width
        y2 = rel_y2 * height
        return self.backend.numpy.concatenate([x1, y1, x2, y2], axis=-1)

    def _rel_yxyx_to_xyxy(self, boxes, height=None, width=None):
        ops = self.backend
        rel_y1, rel_x1, rel_y2, rel_x2 = ops.numpy.split(boxes, 4, axis=-1)
        x1 = rel_x1 * width
        y1 = rel_y1 * height
        x2 = rel_x2 * width
        y2 = rel_y2 * height
        return self.backend.numpy.concatenate([x1, y1, x2, y2], axis=-1)

    def _rel_xywh_to_xyxy(self, boxes, height=None, width=None):
        ops = self.backend
        rel_x1, rel_y1, rel_w, rel_h = ops.numpy.split(boxes, 4, axis=-1)
        x1 = rel_x1 * width
        y1 = rel_y1 * height
        x2 = (rel_x1 + rel_w) * width
        y2 = (rel_y1 + rel_h) * height
        return self.backend.numpy.concatenate([x1, y1, x2, y2], axis=-1)

    def _rel_center_xywh_to_xyxy(self, boxes, height=None, width=None):
        ops = self.backend
        rel_cx, rel_cy, rel_w, rel_h = ops.numpy.split(boxes, 4, axis=-1)
        half_rel_w = rel_w / 2.0
        half_rel_h = rel_h / 2.0
        x1 = (rel_cx - half_rel_w) * height
        y1 = (rel_cy - half_rel_h) * width
        x2 = (rel_cx + half_rel_w) * height
        y2 = (rel_cy + half_rel_h) * width
        return self.backend.numpy.concatenate([x1, y1, x2, y2], axis=-1)

    def _xyxy_to_yxyx(self, boxes, height=None, width=None):
        x1, y1, x2, y2 = self.backend.numpy.split(boxes, 4, axis=-1)
        return self.backend.numpy.concatenate([y1, x1, y2, x2], axis=-1)

    def _xyxy_to_xywh(self, boxes, height=None, width=None):
        x1, y1, x2, y2 = self.backend.numpy.split(boxes, 4, axis=-1)
        w = x2 - x1
        h = y2 - y1
        return self.backend.numpy.concatenate([x1, y1, w, h], axis=-1)

    def _xyxy_to_center_xywh(self, boxes, height=None, width=None):
        x1, y1, x2, y2 = self.backend.numpy.split(boxes, 4, axis=-1)
        cx = x1 + ((x2 - x1) / 2.0)
        cy = y1 + ((y2 - y1) / 2.0)
        w = x2 - x1
        h = y2 - y1
        return self.backend.numpy.concatenate([cx, cy, w, h], axis=-1)

    def _xyxy_to_center_yxhw(self, boxes, height=None, width=None):
        x1, y1, x2, y2 = self.backend.numpy.split(boxes, 4, axis=-1)
        cx = x1 + ((x2 - x1) / 2.0)
        cy = y1 + ((y2 - y1) / 2.0)
        w = x2 - x1
        h = y2 - y1
        return self.backend.numpy.concatenate([cy, cx, h, w], axis=-1)

    def _xyxy_to_rel_xyxy(self, boxes, height=None, width=None):
        x1, y1, x2, y2 = self.backend.numpy.split(boxes, 4, axis=-1)
        rel_x1 = self.backend.numpy.divide(x1, width)
        rel_y1 = self.backend.numpy.divide(y1, height)
        rel_x2 = self.backend.numpy.divide(x2, width)
        rel_y2 = self.backend.numpy.divide(y2, height)
        return self.backend.numpy.concatenate(
            [rel_x1, rel_y1, rel_x2, rel_y2], axis=-1
        )

    def _xyxy_to_rel_yxyx(self, boxes, height=None, width=None):
        x1, y1, x2, y2 = self.backend.numpy.split(boxes, 4, axis=-1)
        rel_x1 = self.backend.numpy.divide(x1, width)
        rel_y1 = self.backend.numpy.divide(y1, height)
        rel_x2 = self.backend.numpy.divide(x2, width)
        rel_y2 = self.backend.numpy.divide(y2, height)
        return self.backend.numpy.concatenate(
            [rel_y1, rel_x1, rel_y2, rel_x2], axis=-1
        )

    def _xyxy_to_rel_xywh(self, boxes, height=None, width=None):
        x1, y1, x2, y2 = self.backend.numpy.split(boxes, 4, axis=-1)
        rel_x1 = x1 / width
        rel_y1 = y1 / height
        rel_w = (x2 - x1) / width
        rel_h = (y2 - y1) / height
        return self.backend.numpy.concatenate(
            [rel_x1, rel_y1, rel_w, rel_h], axis=-1
        )

    def _xyxy_to_rel_center_xywh(self, boxes, height=None, width=None):
        x1, y1, x2, y2 = self.backend.numpy.split(boxes, 4, axis=-1)
        rel_cx = (x1 + ((x2 - x1) / 2.0)) / width
        rel_cy = (y1 + ((y2 - y1) / 2.0)) / height
        rel_w = (x2 - x1) / width
        rel_h = (y2 - y1) / height
        return self.backend.numpy.concatenate(
            [rel_cx, rel_cy, rel_w, rel_h], axis=-1
        )

    # Clip
    def _compute_area(self, boxes, format="xyxy"):
        if format not in ("xyxy", "rel_xyxy"):
            raise NotImplementedError

        ops = self.backend
        x1, y1, x2, y2 = ops.numpy.split(boxes, 4, axis=-1)
        widths = x2 - x1
        heights = y2 - y1
        return widths * heights

    # Affine
    def _compute_inverse_affine_matrix(
        self,
        center_x,
        center_y,
        angle,
        translate_x,
        translate_y,
        scale,
        shear_x,
        shear_y,
        height,
        width,
    ):
        # Ref: TF._geometry._get_inverse_affine_matrix
        ops = self.backend
        batch_size = ops.shape(angle)[0]
        dtype = angle.dtype
        width = ops.cast(width, dtype)
        height = ops.cast(height, dtype)

        angle = -angle
        shear_x = -shear_x
        shear_y = -shear_y

        cx = center_x * width
        cy = center_y * height
        rot = ops.numpy.multiply(angle, 1.0 / 180.0 * math.pi)
        tx = -translate_x * width
        ty = -translate_y * height
        sx = ops.numpy.multiply(shear_x, 1.0 / 180.0 * math.pi)
        sy = ops.numpy.multiply(shear_y, 1.0 / 180.0 * math.pi)

        # Cached results
        cos_sy = ops.numpy.cos(sy)
        tan_sx = ops.numpy.tan(sx)
        rot_minus_sy = rot - sy
        cx_plus_tx = cx + tx
        cy_plus_ty = cy + ty

        # Rotate Scale Shear (RSS) without scaling
        a = ops.numpy.cos(rot_minus_sy) / cos_sy
        b = -(a * tan_sx + ops.numpy.sin(rot))
        c = ops.numpy.sin(rot_minus_sy) / cos_sy
        d = ops.numpy.cos(rot) - c * tan_sx

        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        a0 = d * scale
        a1 = -b * scale
        b0 = -c * scale
        b1 = a * scale
        a2 = cx - a0 * cx_plus_tx - a1 * cy_plus_ty
        b2 = cy - b0 * cx_plus_tx - b1 * cy_plus_ty

        # Shape of matrix: [[batch_size], ...] -> [batch_size, 6]
        matrix = ops.numpy.stack(
            [
                a0,
                a1,
                a2,
                b0,
                b1,
                b2,
                ops.numpy.zeros([batch_size], dtype),
                ops.numpy.zeros([batch_size], dtype),
                ops.numpy.ones([batch_size], dtype),
            ],
            axis=-1,
        )
        matrix = ops.numpy.reshape(matrix, [batch_size, 3, 3])
        return matrix
