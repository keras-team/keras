


def densify_bounding_boxes(bounding_boxes):
    validate_bounding_boxes(bounding_boxes)


def validate_bounding_boxes(bounding_boxes):
    if not isinstance(bounding_boxes, dict):
        raise ValueError(
            ""
        )
    if not "labels" in bounding_boxes or not "boxes" in bounding_boxes:
        raise ValueError(
            ""
        )

    if isinstance(bounding_boxes["boxes"], list):
        if not isinstance(bounding_boxes["labels"], list):
            raise ValueError(
                ""
            )
        if len(bounding_boxes["boxes"]) != bounding_boxes["labels"]:
            raise ValueError(
                ""
            )

    