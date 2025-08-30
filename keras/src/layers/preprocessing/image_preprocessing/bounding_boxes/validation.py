from keras.src import backend as current_backend
from keras.src.utils import tf_utils


def _classes_shape(batched, classes_shape, max_boxes):
    if max_boxes is None:
        return None
    if batched:
        return [None, max_boxes] + classes_shape[2:]
    return [max_boxes] + classes_shape[1:]


def _box_shape(batched, boxes_shape, max_boxes):
    # ensure we dont drop the final axis in RaggedTensor mode
    if max_boxes is None:
        shape = list(boxes_shape)
        shape[-1] = 4
        return shape
    if batched:
        return [None, max_boxes, 4]
    return [max_boxes, 4]


def densify_bounding_boxes(
    bounding_boxes,
    is_batched=False,
    max_boxes=None,
    boxes_default_value=0,
    labels_default_value=-1,
    backend=None,
):
    validate_bounding_boxes(bounding_boxes)
    boxes = bounding_boxes["boxes"]
    labels = bounding_boxes["labels"]
    backend = backend or current_backend
    if isinstance(boxes, list):
        if boxes and isinstance(boxes[0], list):
            if boxes[0] and isinstance(boxes[0][0], list):
                # Batched case
                if not isinstance(labels[0][0], int):
                    raise ValueError(
                        "If providing `bounding_boxes['labels']` as a list, "
                        "it should contain integers labels. Received: "
                        f"bounding_boxes['labels']={labels}"
                    )
                if max_boxes is not None:
                    max_boxes = max([len(b) for b in boxes])
                new_boxes = []
                new_labels = []
                for b, l in zip(boxes, labels):
                    if len(b) >= max_boxes:
                        new_boxes.append(b[:max_boxes])
                        new_labels.append(l[:max_boxes])
                    else:
                        num_boxes_to_add = max_boxes - len(b)
                        added_boxes = [
                            [
                                boxes_default_value,
                                boxes_default_value,
                                boxes_default_value,
                                boxes_default_value,
                            ]
                            for _ in range(num_boxes_to_add)
                        ]
                        new_boxes.append(b + added_boxes)
                        new_labels.append(
                            l
                            + [
                                labels_default_value
                                for _ in range(num_boxes_to_add)
                            ]
                        )
            else:
                # Unbatched case
                if max_boxes and len(b) >= max_boxes:
                    new_boxes = b[:max_boxes]
                    new_labels = l[:max_boxes]
                else:
                    num_boxes_to_add = max_boxes - len(b)
                    added_boxes = [
                        [
                            boxes_default_value,
                            boxes_default_value,
                            boxes_default_value,
                            boxes_default_value,
                        ]
                        for _ in range(num_boxes_to_add)
                    ]
                    new_boxes = b + added_boxes
                    new_labels = l + [
                        labels_default_value for _ in range(num_boxes_to_add)
                    ]
            return {
                "boxes": backend.convert_to_tensor(new_boxes, dtype="float32"),
                "labels": backend.convert_to_tensor(new_labels, dtype="int32"),
            }

    if tf_utils.is_ragged_tensor(boxes):
        bounding_boxes["boxes"] = bounding_boxes["boxes"].to_tensor(
            default_value=boxes_default_value,
            shape=_box_shape(
                is_batched, bounding_boxes["boxes"].shape, max_boxes
            ),
        )
        bounding_boxes["labels"] = bounding_boxes["labels"].to_tensor(
            default_value=labels_default_value,
            shape=_classes_shape(
                is_batched, bounding_boxes["labels"].shape, max_boxes
            ),
        )
        return bounding_boxes

    bounding_boxes["boxes"] = backend.convert_to_tensor(boxes, dtype="float32")
    bounding_boxes["labels"] = backend.convert_to_tensor(labels)
    return bounding_boxes


def validate_bounding_boxes(bounding_boxes):
    if (
        not isinstance(bounding_boxes, dict)
        or "labels" not in bounding_boxes
        or "boxes" not in bounding_boxes
    ):
        raise ValueError(
            "Expected `bounding_boxes` agurment to be a "
            "dict with keys 'boxes' and 'labels'. Received: "
            f"bounding_boxes={bounding_boxes}"
        )
    boxes = bounding_boxes["boxes"]
    labels = bounding_boxes["labels"]
    if isinstance(boxes, list):
        if not isinstance(labels, list):
            raise ValueError(
                "If `bounding_boxes['boxes']` is a list, then "
                "`bounding_boxes['labels']` must also be a list."
                f"Received: bounding_boxes['labels']={labels}"
            )
        if len(boxes) != len(labels):
            raise ValueError(
                "If `bounding_boxes['boxes']` and "
                "`bounding_boxes['labels']` are both lists, "
                "they must have the same length. Received: "
                f"len(bounding_boxes['boxes'])={len(boxes)} and "
                f"len(bounding_boxes['labels'])={len(labels)} and "
            )
    elif tf_utils.is_ragged_tensor(boxes):
        if not tf_utils.is_ragged_tensor(labels):
            raise ValueError(
                "If `bounding_boxes['boxes']` is a Ragged tensor, "
                " `bounding_boxes['labels']` must also be a "
                "Ragged tensor. "
                f"Received: bounding_boxes['labels']={labels}"
            )
    else:
        boxes_shape = current_backend.shape(boxes)
        labels_shape = current_backend.shape(labels)
        if len(boxes_shape) == 2:  # (boxes, 4)
            if len(labels_shape) not in {1, 2}:
                raise ValueError(
                    "Found "
                    f"bounding_boxes['boxes'].shape={boxes_shape} "
                    "and expected bounding_boxes['labels'] to have "
                    "rank 1 or 2, but received: "
                    f"bounding_boxes['labels'].shape={labels_shape} "
                )
        elif len(boxes_shape) == 3:
            if len(labels_shape) not in {2, 3}:
                raise ValueError(
                    "Found "
                    f"bounding_boxes['boxes'].shape={boxes_shape} "
                    "and expected bounding_boxes['labels'] to have "
                    "rank 2 or 3, but received: "
                    f"bounding_boxes['labels'].shape={labels_shape} "
                )
        else:
            raise ValueError(
                "Expected `bounding_boxes['boxes']` "
                "to have rank 2 or 3, with shape "
                "(num_boxes, 4) or (batch_size, num_boxes, 4). "
                "Received: "
                f"bounding_boxes['boxes'].shape={boxes_shape}"
            )
