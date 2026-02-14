import os
import tempfile

from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.visualization.layered_view import layered_view


class LayeredViewTest(testing.TestCase):
    def _build_sequential_cnn(self):
        model = models.Sequential(
            [
                layers.Input(shape=(28, 28, 1)),
                layers.Conv2D(32, 3),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3),
                layers.Flatten(),
                layers.Dense(10),
            ]
        )
        return model

    def _build_functional_model(self):
        inp = layers.Input(shape=(32, 32, 3))
        x = layers.Conv2D(32, 3, padding="same")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        out = layers.Dense(10)(x)
        return models.Model(inputs=inp, outputs=out)

    def test_sequential_model(self):
        model = self._build_sequential_cnn()
        img = layered_view(model)
        self.assertIsNotNone(img)
        self.assertGreater(img.width, 0)
        self.assertGreater(img.height, 0)

    def test_functional_model(self):
        model = self._build_functional_model()
        img = layered_view(model)
        self.assertIsNotNone(img)
        self.assertGreater(img.width, 0)
        self.assertGreater(img.height, 0)

    def test_2d_mode(self):
        model = self._build_sequential_cnn()
        img = layered_view(model, draw_volume=False)
        self.assertIsNotNone(img)

    def test_no_funnel(self):
        model = self._build_sequential_cnn()
        img = layered_view(model, draw_funnel=False)
        self.assertIsNotNone(img)

    def test_custom_color_map(self):
        model = self._build_sequential_cnn()
        custom_colors = {"Conv": (255, 0, 0, 255)}
        img = layered_view(model, color_map=custom_colors)
        self.assertIsNotNone(img)

    def test_legend(self):
        model = self._build_sequential_cnn()
        img_no_legend = layered_view(model, legend=False)
        img_with_legend = layered_view(model, legend=True)
        self.assertGreater(img_with_legend.height, img_no_legend.height)

    def test_text_callable(self):
        model = self._build_sequential_cnn()
        img = layered_view(
            model,
            text_callable=lambda layer: layer.__class__.__name__,
        )
        self.assertIsNotNone(img)

    def test_save_to_file(self):
        model = self._build_sequential_cnn()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.png")
            img = layered_view(model, to_file=filepath)
            self.assertTrue(os.path.exists(filepath))
            self.assertIsNotNone(img)

    def test_unbuilt_model_raises(self):
        model = models.Sequential([layers.Dense(10)])
        with self.assertRaises(ValueError):
            layered_view(model)

    def test_api_access(self):
        import keras

        self.assertTrue(hasattr(keras.visualization, "layered_view"))
