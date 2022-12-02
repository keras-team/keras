"""RetinaNet object detection model.

Adapted from https://keras.io/examples/vision/retinanet/
"""
import tensorflow as tf
from tensorflow import keras

from keras.integration_test.models.input_spec import InputSpec
from keras.saving import serialization_lib

NUM_CLASSES = 10
IMG_SIZE = (224, 224)


def get_data_spec(batch_size):
    return (
        InputSpec((batch_size,) + IMG_SIZE + (3,)),
        InputSpec((batch_size, 9441, 5)),
    )


def get_input_preprocessor():
    return None


def get_backbone():
    backbone = keras.applications.ResNet50(
        include_top=False,
        input_shape=[None, None, 3],
        weights=None,
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in [
            "conv3_block4_out",
            "conv4_block6_out",
            "conv5_block3_out",
        ]
    ]
    return keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )


class FeaturePyramid(keras.layers.Layer):
    def __init__(self, backbone=None, **kwargs):
        super().__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(
            images, training=training
        )
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output


def build_head(output_filters, bias_init):
    head = keras.Sequential([keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(
            keras.layers.Conv2D(
                256, 3, padding="same", kernel_initializer=kernel_init
            )
        )
        head.add(keras.layers.ReLU())
    head.add(
        keras.layers.Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )
    return head


class RetinaNet(keras.Model):
    def __init__(self, num_classes, backbone=None, **kwargs):
        super().__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes

        prior_probability = keras.initializers.Constant(
            -tf.math.log((1 - 0.01) / 0.01)
        )
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "backbone": self.fpn.backbone,
        }

    @classmethod
    def from_config(cls, config):
        backbone = serialization_lib.deserialize_keras_object(
            config.pop("backbone")
        )
        num_classes = config["num_classes"]
        retinanet = cls(num_classes=num_classes, backbone=backbone)
        retinanet(tf.zeros((1, 32, 32, 3)))  # Build model
        return retinanet


class RetinaNetBoxLoss(keras.losses.Loss):
    def __init__(self, delta):
        super().__init__(reduction="none", name="RetinaNetBoxLoss")
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference**2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        return {"delta": self._delta}


class RetinaNetClassificationLoss(keras.losses.Loss):
    def __init__(self, alpha, gamma):
        super().__init__(reduction="none", name="RetinaNetClassificationLoss")
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(
            tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha)
        )
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        return {"alpha": self._alpha, "gamma": self._gamma}


class RetinaNetLoss(keras.losses.Loss):
    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0, delta=1.0):
        super().__init__(reduction="auto", name="RetinaNetLoss")
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes
        self._alpha = alpha
        self._gamma = gamma
        self._delta = delta

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]
        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[:, :, 4:]
        positive_mask = tf.cast(
            tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32
        )
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(
            tf.reduce_sum(clf_loss, axis=-1), normalizer
        )
        box_loss = tf.math.divide_no_nan(
            tf.reduce_sum(box_loss, axis=-1), normalizer
        )
        loss = clf_loss + box_loss
        return loss

    def get_config(self):
        return {
            "num_classes": self._num_classes,
            "alpha": self._alpha,
            "gamma": self._gamma,
            "delta": self._delta,
        }


def get_model(
    build=False, compile=False, jit_compile=False, include_preprocessing=True
):
    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(NUM_CLASSES)
    model = RetinaNet(NUM_CLASSES, resnet50_backbone)

    if compile:
        learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
        learning_rate_boundaries = [125, 250, 500, 240000, 360000]
        learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=learning_rate_boundaries, values=learning_rates
        )
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate_fn, momentum=0.9
        )
        model.compile(
            loss=loss_fn, optimizer=optimizer, jit_compile=jit_compile
        )
    return model


def get_custom_objects():
    return {
        "RetinaNetLoss": RetinaNetLoss,
        "RetinaNetClassificationLoss": RetinaNetClassificationLoss,
        "RetinaNetBoxLoss": RetinaNetBoxLoss,
        "RetinaNet": RetinaNet,
        "FeaturePyramid": FeaturePyramid,
    }
