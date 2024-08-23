from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer

class BaseImagePreprocessingLayer(TFDataLayer):

    _FACTOR_BOUNDS = (-1, 1)

    def _set_factor(self, factor):
        error_msg = (
            "The `factor` argument should be a number "
            "(or a list of two numbers) "
            "in the range "
            f"[{self._FACTOR_BOUNDS[0]}, {self._FACTOR_BOUNDS[1]}]. "
            f"Received: factor={factor}"
        )
        if isinstance(factor, (tuple, list)):
            if len(factor) != 2:
                raise ValueError(error_msg)
            if factor[0] > self._FACTOR_BOUNDS[1] or factor[1] < self._FACTOR_BOUNDS[0]:
                raise ValueError(error_msg)
            lower, upper = sorted(factor)
        elif isinstance(factor, (int, float)):
            if factor < 0 or factor > self._FACTOR_BOUNDS[1]:
                raise ValueError(error_msg)
            factor = abs(factor)
            lower, upper = [max(-factor, self._FACTOR_BOUNDS[0]), factor]
        else:
            raise ValueError(error_msg)
        self.factor_lower = lower
        self.factor_upper = upper

    def get_random_transformation(self, data, seed=None):
        raise NotImplementedError()

    def augment_images(self, images, transformation, **kwargs):
        raise NotImplementedError()

    def augment_targets(self, targets, transformation, **kwargs):
        raise NotImplementedError()

    def augment_bounding_boxes(self, bounding_boxes, transformation, **kwargs):
        raise NotImplementedError()
    
    def augment_segmentation_masks(self, segmentation_masks, transformation, **kwargs):
        raise NotImplementedError()
    
    def call(self, data, training=True):
        if training:
            if self.backend.core.is_tensor(data):
                shape = self.backend.core.shape(data)
                if len(shape) == 3:
                    return self.augment_single_image(data)
                if len(shape) == 4:
                    return self.augment_images(data)
                else:
                    raise ValueError(
                        "Expected image tensor to have rank 3 (single image) "
                        f"or 4 (batch of images). Received: data.shape={shape}"
                    )
            if "images" in data:
                data["images"] = self.augment_images(data["images"])
            else:
                raise ValueError(
                        "Expected data argument to be a tensor of images, or a dict "
                        "containing the key 'images' as well as optional keys "
                        "'bounding_boxes', 'targets', 'segmentation_masks'. Received: "
                        f"data={data}"
                    )
            if "bounding_boxes" in data:
                data["bounding_boxes"] = self.augment_bounding_boxes(data["bounding_boxes"])
            if "targets" in data:
                data["targets"] = self.augment_targets(data["targets"])
        return data