# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for image preprocessing utils."""

import os
import random
import shutil
import tempfile

import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import layers
from keras.engine import sequential
from keras.preprocessing import image
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import image_utils

try:
    import PIL
except ImportError:
    PIL = None


def _generate_test_images(
    include_rgba=False, include_16bit=False, include_32bit=False
):
    img_w = img_h = 20
    rgb_images = []
    rgba_images = []
    gray_images = []
    gray_images_16bit = []
    gray_images_32bit = []
    for _ in range(8):
        bias = np.random.rand(img_w, img_h, 1) * 64
        variance = np.random.rand(img_w, img_h, 1) * (255 - 64)
        # RGB
        imarray = np.random.rand(img_w, img_h, 3) * variance + bias
        im = PIL.Image.fromarray(imarray.astype("uint8")).convert("RGB")
        rgb_images.append(im)
        # RGBA
        imarray = np.random.rand(img_w, img_h, 4) * variance + bias
        im = PIL.Image.fromarray(imarray.astype("uint8")).convert("RGBA")
        rgba_images.append(im)
        # 8-bit grayscale
        imarray = np.random.rand(img_w, img_h, 1) * variance + bias
        im = PIL.Image.fromarray(imarray.astype("uint8").squeeze()).convert("L")
        gray_images.append(im)
        # 16-bit grayscale
        imarray = np.array(
            np.random.randint(-2147483648, 2147483647, (img_w, img_h))
        )
        im = PIL.Image.fromarray(imarray.astype("uint16"))
        gray_images_16bit.append(im)
        # 32-bit grayscale
        im = PIL.Image.fromarray(imarray.astype("uint32"))
        gray_images_32bit.append(im)

    ret = [rgb_images, gray_images]
    if include_rgba:
        ret.append(rgba_images)
    if include_16bit:
        ret.append(gray_images_16bit)
    if include_32bit:
        ret.append(gray_images_32bit)
    return ret


@test_utils.run_v2_only
class TestImage(test_combinations.TestCase):
    def test_iterator_empty_directory(self):
        # Testing with different batch sizes
        for batch_size in [0, 32]:
            data_iterator = image.Iterator(0, batch_size, False, 0)
            ret = next(data_iterator.index_generator)
            self.assertEqual(ret.size, 0)

    def test_image(self):
        if PIL is None:
            return  # Skip test if PIL is not available.

        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                img_list.append(image_utils.img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image.ImageDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=0.0,
                brightness_range=(1, 5),
                fill_mode="nearest",
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
            )
            # Basic test before fit
            x = np.random.random((32, 10, 10, 3))
            generator.flow(x)

            # Fit
            generator.fit(images, augment=True)

            for x, _ in generator.flow(
                images, np.arange(images.shape[0]), shuffle=True
            ):
                self.assertEqual(x.shape[1:], images.shape[1:])
                break

    def test_image_with_split_value_error(self):
        with self.assertRaises(ValueError):
            image.ImageDataGenerator(validation_split=5)

    def test_image_invalid_data(self):
        generator = image.ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format="channels_last",
        )

        # Test fit with invalid data
        with self.assertRaises(ValueError):
            x = np.random.random((3, 10, 10))
            generator.fit(x)
        # Test flow with invalid data
        with self.assertRaises(ValueError):
            generator.flow(np.arange(5))
        # Invalid number of channels: will work but raise a warning
        x = np.random.random((32, 10, 10, 5))
        generator.flow(x)

        with self.assertRaises(ValueError):
            generator = image.ImageDataGenerator(data_format="unknown")

        generator = image.ImageDataGenerator(zoom_range=(2.0, 2.0))

    def test_image_fit(self):
        generator = image.ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format="channels_last",
        )
        # Test grayscale
        x = np.random.random((32, 10, 10, 1))
        generator.fit(x)
        # Test RBG
        x = np.random.random((32, 10, 10, 3))
        generator.fit(x)
        generator = image.ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format="channels_first",
        )
        # Test grayscale
        x = np.random.random((32, 1, 10, 10))
        generator.fit(x)
        # Test RBG
        x = np.random.random((32, 3, 10, 10))
        generator.fit(x)

    def test_directory_iterator(self):
        if PIL is None:
            return  # Skip test if PIL is not available.

        num_classes = 2

        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir)

        # create folders and subfolders
        paths = []
        for cl in range(num_classes):
            class_directory = f"class-{cl}"
            classpaths = [
                class_directory,
                os.path.join(class_directory, "subfolder-1"),
                os.path.join(class_directory, "subfolder-2"),
                os.path.join(class_directory, "subfolder-1", "sub-subfolder"),
            ]
            for path in classpaths:
                os.mkdir(os.path.join(temp_dir, path))
            paths.append(classpaths)

        # save the images in the paths
        count = 0
        filenames = []
        for test_images in _generate_test_images():
            for im in test_images:
                # rotate image class
                im_class = count % num_classes
                # rotate subfolders
                classpaths = paths[im_class]
                filename = os.path.join(
                    classpaths[count % len(classpaths)],
                    f"image-{count}.jpg",
                )
                filenames.append(filename)
                im.save(os.path.join(temp_dir, filename))
                count += 1

        # Test image loading util
        fname = os.path.join(temp_dir, filenames[0])
        _ = image_utils.load_img(fname)
        _ = image_utils.load_img(fname, grayscale=True)
        _ = image_utils.load_img(fname, target_size=(10, 10))
        _ = image_utils.load_img(
            fname, target_size=(10, 10), interpolation="bilinear"
        )

        # create iterator
        generator = image.ImageDataGenerator()
        dir_iterator = generator.flow_from_directory(temp_dir)

        # check number of classes and images
        self.assertEqual(len(dir_iterator.class_indices), num_classes)
        self.assertEqual(len(dir_iterator.classes), count)
        self.assertEqual(set(dir_iterator.filenames), set(filenames))

        def preprocessing_function(x):
            """This will fail if not provided by a Numpy array.

            Note: This is made to enforce backward compatibility.

            Args:
                x: A numpy array.

            Returns:
                An array of zeros with the same shape as the given array.
            """
            self.assertEqual(x.shape, (26, 26, 3))
            self.assertIs(type(x), np.ndarray)
            return np.zeros_like(x)

        # Test usage as Sequence
        generator = image.ImageDataGenerator(
            preprocessing_function=preprocessing_function
        )
        dir_seq = generator.flow_from_directory(
            str(temp_dir),
            target_size=(26, 26),
            color_mode="rgb",
            batch_size=3,
            class_mode="categorical",
        )
        self.assertEqual(len(dir_seq), count // 3 + 1)
        x1, y1 = dir_seq[1]
        self.assertEqual(x1.shape, (3, 26, 26, 3))
        self.assertEqual(y1.shape, (3, num_classes))
        x1, y1 = dir_seq[5]
        self.assertTrue((x1 == 0).all())

    def directory_iterator_with_validation_split_test_helper(
        self, validation_split
    ):
        if PIL is None:
            return  # Skip test if PIL is not available.

        num_classes = 2
        tmp_folder = tempfile.mkdtemp(prefix="test_images")

        # create folders and subfolders
        paths = []
        for cl in range(num_classes):
            class_directory = f"class-{cl}"
            classpaths = [
                class_directory,
                os.path.join(class_directory, "subfolder-1"),
                os.path.join(class_directory, "subfolder-2"),
                os.path.join(class_directory, "subfolder-1", "sub-subfolder"),
            ]
            for path in classpaths:
                os.mkdir(os.path.join(tmp_folder, path))
            paths.append(classpaths)

        # save the images in the paths
        count = 0
        filenames = []
        for test_images in _generate_test_images():
            for im in test_images:
                # rotate image class
                im_class = count % num_classes
                # rotate subfolders
                classpaths = paths[im_class]
                filename = os.path.join(
                    classpaths[count % len(classpaths)],
                    f"image-{count}.jpg",
                )
                filenames.append(filename)
                im.save(os.path.join(tmp_folder, filename))
                count += 1

        # create iterator
        generator = image.ImageDataGenerator(validation_split=validation_split)

        with self.assertRaises(ValueError):
            generator.flow_from_directory(tmp_folder, subset="foo")

        num_validation = int(count * validation_split)
        num_training = count - num_validation
        train_iterator = generator.flow_from_directory(
            tmp_folder, subset="training"
        )
        self.assertEqual(train_iterator.samples, num_training)

        valid_iterator = generator.flow_from_directory(
            tmp_folder, subset="validation"
        )
        self.assertEqual(valid_iterator.samples, num_validation)

        # check number of classes and images
        self.assertEqual(len(train_iterator.class_indices), num_classes)
        self.assertEqual(len(train_iterator.classes), num_training)
        self.assertEqual(
            len(set(train_iterator.filenames) & set(filenames)), num_training
        )

        model = sequential.Sequential([layers.Flatten(), layers.Dense(2)])
        model.compile(optimizer="sgd", loss="mse")
        model.fit(train_iterator, epochs=1)

        shutil.rmtree(tmp_folder)

    @test_combinations.run_all_keras_modes
    def test_directory_iterator_with_validation_split_25_percent(self):
        self.directory_iterator_with_validation_split_test_helper(0.25)

    @test_combinations.run_all_keras_modes
    def test_directory_iterator_with_validation_split_40_percent(self):
        self.directory_iterator_with_validation_split_test_helper(0.40)

    @test_combinations.run_all_keras_modes
    def test_directory_iterator_with_validation_split_50_percent(self):
        self.directory_iterator_with_validation_split_test_helper(0.50)

    def test_batch_standardize(self):
        if PIL is None:
            return  # Skip test if PIL is not available.

        # ImageDataGenerator.standardize should work on batches
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                img_list.append(image_utils.img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image.ImageDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=0.0,
                brightness_range=(1, 5),
                fill_mode="nearest",
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
            )
            generator.fit(images, augment=True)

            transformed = np.copy(images)
            for i, im in enumerate(transformed):
                transformed[i] = generator.random_transform(im)
            transformed = generator.standardize(transformed)

    def test_img_transforms(self):
        x = np.random.random((3, 200, 200))
        _ = image.random_rotation(x, 20)
        _ = image.random_shift(x, 0.2, 0.2)
        _ = image.random_shear(x, 2.0)
        _ = image.random_zoom(x, (0.5, 0.5))
        _ = image.apply_channel_shift(x, 2, 2)
        _ = image.apply_affine_transform(x, 2)
        with self.assertRaises(ValueError):
            image.random_zoom(x, (0, 0, 0))
        _ = image.random_channel_shift(x, 2.0)


@test_utils.run_v2_only
class TestDirectoryIterator(test_combinations.TestCase):
    def test_directory_iterator(self):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(
            include_rgba=True, include_16bit=True, include_32bit=True
        )
        num_classes = 2

        # create folders and subfolders
        paths = []
        for cl in range(num_classes):
            class_directory = f"class-{cl}"
            classpaths = [
                class_directory,
                os.path.join(class_directory, "subfolder-1"),
                os.path.join(class_directory, "subfolder-2"),
                os.path.join(class_directory, "subfolder-1", "sub-subfolder"),
            ]
            for path in classpaths:
                os.mkdir(os.path.join(tmpdir.full_path, path))
            paths.append(classpaths)

        # save the images in the paths
        count = 0
        filenames = []
        for test_images in all_test_images:
            for im in test_images:
                # rotate image class
                im_class = count % num_classes
                # rotate subfolders
                classpaths = paths[im_class]
                filename = os.path.join(
                    classpaths[count % len(classpaths)],
                    f"image-{count}.png",
                )
                filenames.append(filename)
                im.save(os.path.join(tmpdir.full_path, filename))
                count += 1

        # create iterator
        generator = image.ImageDataGenerator()
        dir_iterator = generator.flow_from_directory(tmpdir.full_path)

        # check number of classes and images
        self.assertLen(dir_iterator.class_indices, num_classes)
        self.assertLen(dir_iterator.classes, count)
        self.assertEqual(set(dir_iterator.filenames), set(filenames))

        # Test invalid use cases
        with self.assertRaises(ValueError):
            generator.flow_from_directory(tmpdir.full_path, color_mode="cmyk")
        with self.assertRaises(ValueError):
            generator.flow_from_directory(tmpdir.full_path, class_mode="output")

        def preprocessing_function(x):
            # This will fail if not provided by a Numpy array.
            # Note: This is made to enforce backward compatibility.
            self.assertEqual(x.shape, (26, 26, 3))
            self.assertIsInstance(x, np.ndarray)

            return np.zeros_like(x)

        # Test usage as Sequence
        generator = image.ImageDataGenerator(
            preprocessing_function=preprocessing_function
        )
        dir_seq = generator.flow_from_directory(
            tmpdir.full_path,
            target_size=(26, 26),
            color_mode="rgb",
            batch_size=3,
            class_mode="categorical",
        )
        self.assertLen(dir_seq, np.ceil(count / 3.0))
        x1, y1 = dir_seq[1]
        self.assertEqual(x1.shape, (3, 26, 26, 3))
        self.assertEqual(y1.shape, (3, num_classes))
        x1, y1 = dir_seq[5]
        self.assertTrue((x1 == 0).all())

        with self.assertRaises(ValueError):
            x1, y1 = dir_seq[14]  # there are 40 images and batch size is 3

    def test_directory_iterator_class_mode_input(self):
        tmpdir = self.create_tempdir()
        os.mkdir(os.path.join(tmpdir.full_path, "class-1"))
        all_test_images = _generate_test_images(
            include_rgba=True, include_16bit=True, include_32bit=True
        )

        # save the images in the paths
        count = 0
        for test_images in all_test_images:
            for im in test_images:
                filename = os.path.join(tmpdir, "class-1", f"image-{count}.png")
                im.save(filename)
                count += 1

        # create iterator
        generator = image.ImageDataGenerator()
        dir_iterator = generator.flow_from_directory(
            tmpdir.full_path, class_mode="input"
        )
        batch = next(dir_iterator)

        # check if input and output have the same shape
        self.assertEqual(batch[0].shape, batch[1].shape)
        # check if the input and output images are not the same numpy array
        input_img = batch[0][0]
        output_img = batch[1][0]
        output_img[0][0][0] += 1
        self.assertNotEqual(input_img[0][0][0], output_img[0][0][0])

    @parameterized.parameters(
        [
            (0.25, 30),
            (0.50, 20),
            (0.75, 10),
        ]
    )
    def test_directory_iterator_with_validation_split(
        self, validation_split, num_training
    ):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(
            include_rgba=True, include_16bit=True, include_32bit=True
        )
        num_classes = 2

        # create folders and subfolders
        paths = []
        for cl in range(num_classes):
            class_directory = f"class-{cl}"
            classpaths = [
                class_directory,
                os.path.join(class_directory, "subfolder-1"),
                os.path.join(class_directory, "subfolder-2"),
                os.path.join(class_directory, "subfolder-1", "sub-subfolder"),
            ]
            for path in classpaths:
                os.mkdir(os.path.join(tmpdir.full_path, path))
            paths.append(classpaths)

        # save the images in the paths
        count = 0
        filenames = []
        for test_images in all_test_images:
            for im in test_images:
                # rotate image class
                im_class = count % num_classes
                # rotate subfolders
                classpaths = paths[im_class]
                filename = os.path.join(
                    classpaths[count % len(classpaths)],
                    f"image-{count}.png",
                )
                filenames.append(filename)
                im.save(os.path.join(tmpdir.full_path, filename))
                count += 1

        # create iterator
        generator = image.ImageDataGenerator(validation_split=validation_split)

        with self.assertRaises(ValueError):
            generator.flow_from_directory(tmpdir.full_path, subset="foo")

        train_iterator = generator.flow_from_directory(
            tmpdir.full_path, subset="training"
        )
        self.assertEqual(train_iterator.samples, num_training)

        valid_iterator = generator.flow_from_directory(
            tmpdir.full_path, subset="validation"
        )
        self.assertEqual(valid_iterator.samples, count - num_training)

        # check number of classes and images
        self.assertLen(train_iterator.class_indices, num_classes)
        self.assertLen(train_iterator.classes, num_training)
        self.assertLen(
            set(train_iterator.filenames) & set(filenames), num_training
        )


@test_utils.run_v2_only
class TestNumpyArrayIterator(test_combinations.TestCase):
    def test_numpy_array_iterator(self):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(include_rgba=True)

        image_data_generator = image.ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            rotation_range=90.0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.5,
            zoom_range=0.2,
            channel_shift_range=0.0,
            brightness_range=(1, 5),
            fill_mode="nearest",
            cval=0.5,
            horizontal_flip=True,
            vertical_flip=True,
            interpolation_order=1,
        )

        for test_images in all_test_images:
            img_list = []
            for im in test_images:
                img_list.append(image_utils.img_to_array(im)[None, ...])
            images = np.vstack(img_list)
            dsize = images.shape[0]

            iterator = image.NumpyArrayIterator(
                images,
                np.arange(images.shape[0]),
                image_data_generator,
                shuffle=False,
                save_to_dir=tmpdir.full_path,
                batch_size=3,
            )
            x, y = next(iterator)
            self.assertEqual(x.shape, images[:3].shape)
            self.assertEqual(list(y), [0, 1, 2])

            # Test with sample weights
            iterator = image.NumpyArrayIterator(
                images,
                np.arange(images.shape[0]),
                image_data_generator,
                shuffle=False,
                sample_weight=np.arange(images.shape[0]) + 1,
                save_to_dir=tmpdir.full_path,
                batch_size=3,
            )
            x, y, w = iterator.next()
            self.assertEqual(x.shape, images[:3].shape)
            self.assertEqual(list(y), [0, 1, 2])
            self.assertEqual(list(w), [1, 2, 3])

            # Test with `shuffle=True`
            iterator = image.NumpyArrayIterator(
                images,
                np.arange(images.shape[0]),
                image_data_generator,
                shuffle=True,
                save_to_dir=tmpdir.full_path,
                batch_size=3,
                seed=42,
            )
            x, y = iterator.next()
            self.assertEqual(x.shape, images[:3].shape)
            # Check that the sequence is shuffled.
            self.assertNotEqual(list(y), [0, 1, 2])

            # Test without y
            iterator = image.NumpyArrayIterator(
                images,
                None,
                image_data_generator,
                shuffle=True,
                save_to_dir=tmpdir.full_path,
                batch_size=3,
            )
            x = iterator.next()
            self.assertIsInstance(x, np.ndarray)
            self.assertEqual(x.shape, images[:3].shape)

            # Test with a single miscellaneous input data array
            x_misc1 = np.random.random(dsize)
            iterator = image.NumpyArrayIterator(
                (images, x_misc1),
                np.arange(dsize),
                image_data_generator,
                shuffle=False,
                batch_size=2,
            )
            for i, (x, y) in enumerate(iterator):
                self.assertEqual(x[0].shape, images[:2].shape)
                self.assertTrue(
                    (x[1] == x_misc1[(i * 2) : ((i + 1) * 2)]).all()
                )
                if i == 2:
                    break

            # Test with two miscellaneous inputs
            x_misc2 = np.random.random((dsize, 3, 3))
            iterator = image.NumpyArrayIterator(
                (images, [x_misc1, x_misc2]),
                np.arange(dsize),
                image_data_generator,
                shuffle=False,
                batch_size=2,
            )
            for i, (x, y) in enumerate(iterator):
                self.assertEqual(x[0].shape, images[:2].shape)
                self.assertTrue(
                    (x[1] == x_misc1[(i * 2) : ((i + 1) * 2)]).all()
                )
                self.assertTrue(
                    (x[2] == x_misc2[(i * 2) : ((i + 1) * 2)]).all()
                )
                if i == 2:
                    break

            # Test cases with `y = None`
            iterator = image.NumpyArrayIterator(
                images, None, image_data_generator, batch_size=3
            )
            x = iterator.next()
            self.assertIsInstance(x, np.ndarray)
            self.assertEqual(x.shape, images[:3].shape)

            iterator = image.NumpyArrayIterator(
                (images, x_misc1),
                None,
                image_data_generator,
                batch_size=3,
                shuffle=False,
            )
            x = iterator.next()
            self.assertIsInstance(x, list)
            self.assertEqual(x[0].shape, images[:3].shape)
            self.assertTrue((x[1] == x_misc1[:3]).all())

            iterator = image.NumpyArrayIterator(
                (images, [x_misc1, x_misc2]),
                None,
                image_data_generator,
                batch_size=3,
                shuffle=False,
            )
            x = iterator.next()
            self.assertIsInstance(x, list)
            self.assertEqual(x[0].shape, images[:3].shape)
            self.assertTrue((x[1] == x_misc1[:3]).all())
            self.assertTrue((x[2] == x_misc2[:3]).all())

            # Test with validation split
            generator = image.ImageDataGenerator(validation_split=0.2)
            iterator = image.NumpyArrayIterator(
                images, None, generator, batch_size=3
            )
            x = iterator.next()
            self.assertIsInstance(x, np.ndarray)
            self.assertEqual(x.shape, images[:3].shape)

            # Test some failure cases:
            x_misc_err = np.random.random((dsize + 1, 3, 3))

            with self.assertRaisesRegex(ValueError, "All of the arrays in"):
                image.NumpyArrayIterator(
                    (images, x_misc_err),
                    np.arange(dsize),
                    generator,
                    batch_size=3,
                )

            with self.assertRaisesRegex(
                ValueError, r"`x` \(images tensor\) and `y` \(labels\)"
            ):
                image.NumpyArrayIterator(
                    (images, x_misc1),
                    np.arange(dsize + 1),
                    generator,
                    batch_size=3,
                )

            # Test `flow` behavior as Sequence
            seq = image.NumpyArrayIterator(
                images,
                np.arange(images.shape[0]),
                generator,
                shuffle=False,
                save_to_dir=tmpdir.full_path,
                batch_size=3,
            )
            self.assertLen(seq, images.shape[0] // 3 + 1)
            x, y = seq[0]
            self.assertEqual(x.shape, images[:3].shape)
            self.assertEqual(list(y), [0, 1, 2])

            # Test with `shuffle=True`
            seq = image.NumpyArrayIterator(
                images,
                np.arange(images.shape[0]),
                generator,
                shuffle=True,
                save_to_dir=tmpdir.full_path,
                batch_size=3,
                seed=123,
            )
            x, y = seq[0]
            # Check that the sequence is shuffled.
            self.assertNotEqual(list(y), [0, 1, 2])
            # `on_epoch_end` should reshuffle the sequence.
            seq.on_epoch_end()
            _, y2 = seq[0]
            self.assertNotEqual(list(y), list(y2))

        # test order_interpolation
        labels = np.array(
            [
                [2, 2, 0, 2, 2],
                [1, 3, 2, 3, 1],
                [2, 1, 0, 1, 2],
                [3, 1, 0, 2, 0],
                [3, 1, 3, 2, 1],
            ]
        )
        label_generator = image.ImageDataGenerator(
            rotation_range=90.0, interpolation_order=0
        )
        labels_gen = image.NumpyArrayIterator(
            labels[np.newaxis, ..., np.newaxis], None, label_generator, seed=123
        )
        self.assertTrue(
            (np.unique(labels) == np.unique(next(labels_gen))).all()
        )


@test_utils.run_v2_only
class TestDataFrameIterator(test_combinations.TestCase):
    def test_dataframe_iterator(self):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(include_rgba=True)
        num_classes = 2

        # save the images in the tmpdir
        count = 0
        filenames = []
        filepaths = []
        filenames_without = []
        for test_images in all_test_images:
            for im in test_images:
                filename = f"image-{count}.png"
                filename_without = f"image-{count}"
                filenames.append(filename)
                filepaths.append(os.path.join(tmpdir.full_path, filename))
                filenames_without.append(filename_without)
                im.save(os.path.join(tmpdir.full_path, filename))
                count += 1

        df = pd.DataFrame(
            {
                "filename": filenames,
                "class": [str(random.randint(0, 1)) for _ in filenames],
                "filepaths": filepaths,
            }
        )

        # create iterator
        iterator = image.DataFrameIterator(df, tmpdir.full_path)
        batch = next(iterator)
        self.assertLen(batch, 2)
        self.assertIsInstance(batch[0], np.ndarray)
        self.assertIsInstance(batch[1], np.ndarray)
        generator = image.ImageDataGenerator()
        df_iterator = generator.flow_from_dataframe(df, x_col="filepaths")
        df_iterator_dir = generator.flow_from_dataframe(df, tmpdir.full_path)
        df_sparse_iterator = generator.flow_from_dataframe(
            df, tmpdir.full_path, class_mode="sparse"
        )
        self.assertFalse(np.isnan(df_sparse_iterator.classes).any())
        # check number of classes and images
        self.assertLen(df_iterator.class_indices, num_classes)
        self.assertLen(df_iterator.classes, count)
        self.assertEqual(set(df_iterator.filenames), set(filepaths))
        self.assertLen(df_iterator_dir.class_indices, num_classes)
        self.assertLen(df_iterator_dir.classes, count)
        self.assertEqual(set(df_iterator_dir.filenames), set(filenames))
        # test without shuffle
        _, batch_y = next(
            generator.flow_from_dataframe(
                df, tmpdir.full_path, shuffle=False, class_mode="sparse"
            )
        )
        self.assertTrue(
            (batch_y == df["class"].astype("float")[: len(batch_y)]).all()
        )
        # Test invalid use cases
        with self.assertRaises(ValueError):
            generator.flow_from_dataframe(
                df, tmpdir.full_path, color_mode="cmyk"
            )
        with self.assertRaises(ValueError):
            generator.flow_from_dataframe(
                df, tmpdir.full_path, class_mode="output"
            )
        with self.assertWarns(DeprecationWarning):
            generator.flow_from_dataframe(df, tmpdir.full_path, has_ext=True)
        with self.assertWarns(DeprecationWarning):
            generator.flow_from_dataframe(df, tmpdir.full_path, has_ext=False)

        def preprocessing_function(x):
            # This will fail if not provided by a Numpy array.
            # Note: This is made to enforce backward compatibility.

            self.assertEqual(x.shape, (26, 26, 3))
            self.assertIsInstance(x, np.ndarray)

            return np.zeros_like(x)

        # Test usage as Sequence
        generator = image.ImageDataGenerator(
            preprocessing_function=preprocessing_function
        )
        dir_seq = generator.flow_from_dataframe(
            df,
            tmpdir.full_path,
            target_size=(26, 26),
            color_mode="rgb",
            batch_size=3,
            class_mode="categorical",
        )
        self.assertLen(dir_seq, np.ceil(count / 3))
        x1, y1 = dir_seq[1]
        self.assertEqual(x1.shape, (3, 26, 26, 3))
        self.assertEqual(y1.shape, (3, num_classes))
        x1, y1 = dir_seq[5]
        self.assertTrue((x1 == 0).all())

        with self.assertRaises(ValueError):
            x1, y1 = dir_seq[9]

    def test_dataframe_iterator_validate_filenames(self):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(include_rgba=True)
        # save the images in the paths
        count = 0
        filenames = []
        for test_images in all_test_images:
            for im in test_images:
                filename = f"image-{count}.png"
                im.save(os.path.join(tmpdir.full_path, filename))
                filenames.append(filename)
                count += 1
        df = pd.DataFrame({"filename": filenames + ["test.jpp", "test.jpg"]})
        generator = image.ImageDataGenerator()
        df_iterator = generator.flow_from_dataframe(
            df, tmpdir.full_path, class_mode="input"
        )
        self.assertLen(df_iterator.filenames, len(df["filename"]) - 2)
        df_iterator = generator.flow_from_dataframe(
            df, tmpdir.full_path, class_mode="input", validate_filenames=False
        )
        self.assertLen(df_iterator.filenames, len(df["filename"]))

    def test_dataframe_iterator_sample_weights(self):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(include_rgba=True)
        # save the images in the paths
        count = 0
        filenames = []
        for test_images in all_test_images:
            for im in test_images:
                filename = f"image-{count}.png"
                im.save(os.path.join(tmpdir.full_path, filename))
                filenames.append(filename)
                count += 1
        df = pd.DataFrame({"filename": filenames})
        df["weight"] = ([2, 5] * len(df))[: len(df)]
        generator = image.ImageDataGenerator()
        df_iterator = generator.flow_from_dataframe(
            df,
            tmpdir.full_path,
            x_col="filename",
            y_col=None,
            shuffle=False,
            batch_size=5,
            weight_col="weight",
            class_mode="input",
        )

        batch = next(df_iterator)
        self.assertLen(batch, 3)  # (x, y, weights)
        # check if input and output have the same shape and they're the same
        self.assertEqual(batch[0].all(), batch[1].all())
        # check if the input and output images are not the same numpy array
        input_img = batch[0][0]
        output_img = batch[1][0]
        output_img[0][0][0] += 1
        self.assertNotEqual(input_img[0][0][0], output_img[0][0][0])
        self.assertAllEqual(np.array([2, 5, 2, 5, 2]), batch[2])

        # fail
        df["weight"] = (["2", "5"] * len(df))[: len(df)]
        with self.assertRaises(TypeError):
            image.ImageDataGenerator().flow_from_dataframe(
                df, weight_col="weight", class_mode="input"
            )

    def test_dataframe_iterator_class_mode_input(self):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(include_rgba=True)
        # save the images in the paths
        count = 0
        filenames = []
        for test_images in all_test_images:
            for im in test_images:
                filename = f"image-{count}.png"
                im.save(os.path.join(tmpdir.full_path, filename))
                filenames.append(filename)
                count += 1
        df = pd.DataFrame({"filename": filenames})
        generator = image.ImageDataGenerator()
        df_autoencoder_iterator = generator.flow_from_dataframe(
            df,
            tmpdir.full_path,
            x_col="filename",
            y_col=None,
            class_mode="input",
        )

        batch = next(df_autoencoder_iterator)

        # check if input and output have the same shape and they're the same
        self.assertAllClose(batch[0], batch[1])
        # check if the input and output images are not the same numpy array
        input_img = batch[0][0]
        output_img = batch[1][0]
        output_img[0][0][0] += 1
        self.assertNotEqual(input_img[0][0][0], output_img[0][0][0])

        df_autoencoder_iterator = generator.flow_from_dataframe(
            df,
            tmpdir.full_path,
            x_col="filename",
            y_col="class",
            class_mode="input",
        )

        batch = next(df_autoencoder_iterator)

        # check if input and output have the same shape and they're the same
        self.assertEqual(batch[0].all(), batch[1].all())
        # check if the input and output images are not the same numpy array
        input_img = batch[0][0]
        output_img = batch[1][0]
        output_img[0][0][0] += 1
        self.assertNotEqual(input_img[0][0][0], output_img[0][0][0])

    def test_dataframe_iterator_class_mode_categorical_multi_label(self):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(include_rgba=True)
        # save the images in the paths
        filenames = []
        count = 0
        for test_images in all_test_images:
            for im in test_images:
                filename = f"image-{count}.png"
                im.save(os.path.join(tmpdir.full_path, filename))
                filenames.append(filename)
                count += 1
        label_opt = ["a", "b", ["a"], ["b"], ["a", "b"], ["b", "a"]]
        df = pd.DataFrame(
            {
                "filename": filenames,
                "class": [random.choice(label_opt) for _ in filenames[:-2]]
                + ["b", "a"],
            }
        )
        generator = image.ImageDataGenerator()
        df_iterator = generator.flow_from_dataframe(df, tmpdir.full_path)
        batch_x, batch_y = next(df_iterator)
        self.assertIsInstance(batch_x, np.ndarray)
        self.assertLen(batch_x.shape, 4)
        self.assertIsInstance(batch_y, np.ndarray)
        self.assertEqual(batch_y.shape, (len(batch_x), 2))
        for labels in batch_y:
            self.assertTrue(all(label in {0, 1} for label in labels))

        # on first 3 batches
        df = pd.DataFrame(
            {
                "filename": filenames,
                "class": [["b", "a"]]
                + ["b"]
                + [["c"]]
                + [random.choice(label_opt) for _ in filenames[:-3]],
            }
        )
        generator = image.ImageDataGenerator()
        df_iterator = generator.flow_from_dataframe(
            df, tmpdir.full_path, shuffle=False
        )
        batch_x, batch_y = next(df_iterator)
        self.assertIsInstance(batch_x, np.ndarray)
        self.assertLen(batch_x.shape, 4)
        self.assertIsInstance(batch_y, np.ndarray)
        self.assertEqual(batch_y.shape, (len(batch_x), 3))
        for labels in batch_y:
            self.assertTrue(all(label in {0, 1} for label in labels))
        self.assertTrue((batch_y[0] == np.array([1, 1, 0])).all())
        self.assertTrue((batch_y[1] == np.array([0, 1, 0])).all())
        self.assertTrue((batch_y[2] == np.array([0, 0, 1])).all())

    def test_dataframe_iterator_class_mode_multi_output(self):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(include_rgba=True)
        # save the images in the paths
        filenames = []
        count = 0
        for test_images in all_test_images:
            for im in test_images:
                filename = f"image-{count}.png"
                im.save(os.path.join(tmpdir.full_path, filename))
                filenames.append(filename)
                count += 1
        # fit both outputs are a single number
        df = pd.DataFrame({"filename": filenames}).assign(
            output_0=np.random.uniform(size=len(filenames)),
            output_1=np.random.uniform(size=len(filenames)),
        )
        df_iterator = image.ImageDataGenerator().flow_from_dataframe(
            df,
            y_col=["output_0", "output_1"],
            directory=tmpdir.full_path,
            batch_size=3,
            shuffle=False,
            class_mode="multi_output",
        )
        batch_x, batch_y = next(df_iterator)
        self.assertIsInstance(batch_x, np.ndarray)
        self.assertLen(batch_x.shape, 4)
        self.assertIsInstance(batch_y, list)
        self.assertLen(batch_y, 2)
        self.assertAllEqual(batch_y[0], np.array(df["output_0"].tolist()[:3]))
        self.assertAllEqual(batch_y[1], np.array(df["output_1"].tolist()[:3]))
        # if one of the outputs is a 1D array
        df["output_1"] = [
            np.random.uniform(size=(2, 2, 1)).flatten() for _ in range(len(df))
        ]
        df_iterator = image.ImageDataGenerator().flow_from_dataframe(
            df,
            y_col=["output_0", "output_1"],
            directory=tmpdir.full_path,
            batch_size=3,
            shuffle=False,
            class_mode="multi_output",
        )
        batch_x, batch_y = next(df_iterator)
        self.assertIsInstance(batch_x, np.ndarray)
        self.assertLen(batch_x.shape, 4)
        self.assertIsInstance(batch_y, list)
        self.assertLen(batch_y, 2)
        self.assertAllEqual(batch_y[0], np.array(df["output_0"].tolist()[:3]))
        self.assertAllEqual(batch_y[1], np.array(df["output_1"].tolist()[:3]))
        # if one of the outputs is a 2D array
        df["output_1"] = [
            np.random.uniform(size=(2, 2, 1)) for _ in range(len(df))
        ]
        df_iterator = image.ImageDataGenerator().flow_from_dataframe(
            df,
            y_col=["output_0", "output_1"],
            directory=tmpdir.full_path,
            batch_size=3,
            shuffle=False,
            class_mode="multi_output",
        )
        batch_x, batch_y = next(df_iterator)
        self.assertIsInstance(batch_x, np.ndarray)
        self.assertLen(batch_x.shape, 4)
        self.assertIsInstance(batch_y, list)
        self.assertLen(batch_y, 2)
        self.assertAllEqual(batch_y[0], np.array(df["output_0"].tolist()[:3]))
        self.assertAllEqual(batch_y[1], np.array(df["output_1"].tolist()[:3]))
        # fail if single column
        with self.assertRaises(TypeError):
            image.ImageDataGenerator().flow_from_dataframe(
                df,
                y_col="output_0",
                directory=tmpdir.full_path,
                class_mode="multi_output",
            )

    def test_dataframe_iterator_class_mode_raw(self):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(include_rgba=True)
        # save the images in the paths
        filenames = []
        count = 0
        for test_images in all_test_images:
            for im in test_images:
                filename = f"image-{count}.png"
                im.save(os.path.join(tmpdir.full_path, filename))
                filenames.append(filename)
                count += 1
        # case for 1D output
        df = pd.DataFrame({"filename": filenames}).assign(
            output_0=np.random.uniform(size=len(filenames)),
            output_1=np.random.uniform(size=len(filenames)),
        )
        df_iterator = image.ImageDataGenerator().flow_from_dataframe(
            df,
            y_col="output_0",
            directory=tmpdir.full_path,
            batch_size=3,
            shuffle=False,
            class_mode="raw",
        )
        batch_x, batch_y = next(df_iterator)
        self.assertIsInstance(batch_x, np.ndarray)
        self.assertLen(batch_x.shape, 4)
        self.assertIsInstance(batch_y, np.ndarray)
        self.assertEqual(batch_y.shape, (3,))
        self.assertAllEqual(batch_y, df["output_0"].values[:3])
        # case with a 2D output
        df_iterator = image.ImageDataGenerator().flow_from_dataframe(
            df,
            y_col=["output_0", "output_1"],
            directory=tmpdir.full_path,
            batch_size=3,
            shuffle=False,
            class_mode="raw",
        )
        batch_x, batch_y = next(df_iterator)
        self.assertIsInstance(batch_x, np.ndarray)
        self.assertLen(batch_x.shape, 4)
        self.assertIsInstance(batch_y, np.ndarray)
        self.assertEqual(batch_y.shape, (3, 2))
        self.assertAllEqual(batch_y, df[["output_0", "output_1"]].values[:3])

    @parameterized.parameters(
        [
            (0.25, 18),
            (0.50, 12),
            (0.75, 6),
        ]
    )
    def test_dataframe_iterator_with_validation_split(
        self, validation_split, num_training
    ):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(include_rgba=True)
        num_classes = 2

        # save the images in the tmpdir
        count = 0
        filenames = []
        filenames_without = []
        for test_images in all_test_images:
            for im in test_images:
                filename = f"image-{count}.png"
                filename_without = f"image-{count}"
                filenames.append(filename)
                filenames_without.append(filename_without)
                im.save(os.path.join(tmpdir.full_path, filename))
                count += 1

        df = pd.DataFrame(
            {
                "filename": filenames,
                "class": [str(random.randint(0, 1)) for _ in filenames],
            }
        )
        # create iterator
        generator = image.ImageDataGenerator(validation_split=validation_split)
        df_sparse_iterator = generator.flow_from_dataframe(
            df, tmpdir.full_path, class_mode="sparse"
        )
        if np.isnan(next(df_sparse_iterator)[:][1]).any():
            raise ValueError("Invalid values.")

        with self.assertRaises(ValueError):
            generator.flow_from_dataframe(df, tmpdir.full_path, subset="foo")

        train_iterator = generator.flow_from_dataframe(
            df, tmpdir.full_path, subset="training"
        )
        self.assertEqual(train_iterator.samples, num_training)

        valid_iterator = generator.flow_from_dataframe(
            df, tmpdir.full_path, subset="validation"
        )
        self.assertEqual(valid_iterator.samples, count - num_training)

        # check number of classes and images
        self.assertLen(train_iterator.class_indices, num_classes)
        self.assertLen(train_iterator.classes, num_training)
        self.assertLen(
            set(train_iterator.filenames) & set(filenames), num_training
        )

    def test_dataframe_iterator_with_custom_indexed_dataframe(self):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(include_rgba=True)
        num_classes = 2

        # save the images in the tmpdir
        count = 0
        filenames = []
        for test_images in all_test_images:
            for im in test_images:
                filename = f"image-{count}.png"
                filenames.append(filename)
                im.save(os.path.join(tmpdir.full_path, filename))
                count += 1

        # create dataframes
        classes = np.random.randint(num_classes, size=len(filenames))
        classes = [str(c) for c in classes]
        df = pd.DataFrame({"filename": filenames, "class": classes})
        df2 = pd.DataFrame(
            {"filename": filenames, "class": classes},
            index=np.arange(1, len(filenames) + 1),
        )
        df3 = pd.DataFrame(
            {"filename": filenames, "class": classes}, index=filenames
        )

        # create iterators
        seed = 1
        generator = image.ImageDataGenerator()
        df_iterator = generator.flow_from_dataframe(
            df, tmpdir.full_path, seed=seed
        )
        df2_iterator = generator.flow_from_dataframe(
            df2, tmpdir.full_path, seed=seed
        )
        df3_iterator = generator.flow_from_dataframe(
            df3, tmpdir.full_path, seed=seed
        )

        # Test all iterators return same pairs of arrays
        for _ in range(len(filenames)):
            a1, c1 = next(df_iterator)
            a2, c2 = next(df2_iterator)
            a3, c3 = next(df3_iterator)
            self.assertAllEqual(a1, a2)
            self.assertAllEqual(a1, a3)
            self.assertAllEqual(c1, c2)
            self.assertAllEqual(c1, c3)

    def test_dataframe_iterator_n(self):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(include_rgba=True)

        # save the images in the tmpdir
        count = 0
        filenames = []
        for test_images in all_test_images:
            for im in test_images:
                filename = f"image-{count}.png"
                filenames.append(filename)
                im.save(os.path.join(tmpdir.full_path, filename))
                count += 1

        # exclude first two items
        n_files = len(filenames)
        input_filenames = filenames[2:]

        # create dataframes
        classes = np.random.randint(2, size=len(input_filenames))
        classes = [str(c) for c in classes]
        df = pd.DataFrame({"filename": input_filenames})
        df2 = pd.DataFrame({"filename": input_filenames, "class": classes})

        # create iterators
        generator = image.ImageDataGenerator()
        df_iterator = generator.flow_from_dataframe(
            df, tmpdir.full_path, class_mode=None
        )
        df2_iterator = generator.flow_from_dataframe(
            df2, tmpdir.full_path, class_mode="binary"
        )

        # Test the number of items in iterators
        self.assertEqual(df_iterator.n, n_files - 2)
        self.assertEqual(df2_iterator.n, n_files - 2)

    def test_dataframe_iterator_absolute_path(self):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(include_rgba=True)

        # save the images in the tmpdir
        count = 0
        file_paths = []
        for test_images in all_test_images:
            for im in test_images:
                filename = f"image-{count:0>5}.png"
                file_path = os.path.join(tmpdir.full_path, filename)
                file_paths.append(file_path)
                im.save(file_path)
                count += 1

        # prepare an image with a forbidden extension.
        file_path_fbd = os.path.join(tmpdir.full_path, "image-forbid.fbd")
        shutil.copy(file_path, file_path_fbd)

        # create dataframes
        classes = np.random.randint(2, size=len(file_paths))
        classes = [str(c) for c in classes]
        df = pd.DataFrame({"filename": file_paths})
        df2 = pd.DataFrame({"filename": file_paths, "class": classes})
        df3 = pd.DataFrame({"filename": ["image-not-exist.png"] + file_paths})
        df4 = pd.DataFrame({"filename": file_paths + [file_path_fbd]})

        # create iterators
        generator = image.ImageDataGenerator()
        df_iterator = generator.flow_from_dataframe(
            df, None, class_mode=None, shuffle=False, batch_size=1
        )
        df2_iterator = generator.flow_from_dataframe(
            df2, None, class_mode="binary", shuffle=False, batch_size=1
        )
        df3_iterator = generator.flow_from_dataframe(
            df3, None, class_mode=None, shuffle=False, batch_size=1
        )
        df4_iterator = generator.flow_from_dataframe(
            df4, None, class_mode=None, shuffle=False, batch_size=1
        )

        validation_split = 0.2
        generator_split = image.ImageDataGenerator(
            validation_split=validation_split
        )
        df_train_iterator = generator_split.flow_from_dataframe(
            df,
            None,
            class_mode=None,
            shuffle=False,
            subset="training",
            batch_size=1,
        )
        df_val_iterator = generator_split.flow_from_dataframe(
            df,
            None,
            class_mode=None,
            shuffle=False,
            subset="validation",
            batch_size=1,
        )

        # Test the number of items in iterators
        self.assertLen(file_paths, df_iterator.n)
        self.assertLen(file_paths, df2_iterator.n)
        self.assertLen(file_paths, df3_iterator.n)
        self.assertLen(file_paths, df4_iterator.n)
        self.assertEqual(
            df_val_iterator.n, int(validation_split * len(file_paths))
        )
        self.assertLen(file_paths, df_train_iterator.n + df_val_iterator.n)

        # Test flow_from_dataframe
        for i in range(len(file_paths)):
            a1 = next(df_iterator)
            a2, _ = next(df2_iterator)
            a3 = next(df3_iterator)
            a4 = next(df4_iterator)

            if i < df_val_iterator.n:
                a5 = next(df_val_iterator)
            else:
                a5 = next(df_train_iterator)

            self.assertAllEqual(a1, a2)
            self.assertAllEqual(a1, a3)
            self.assertAllEqual(a1, a4)
            self.assertAllEqual(a1, a5)

    def test_dataframe_iterator_with_subdirs(self):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(include_rgba=True)
        num_classes = 2

        # create folders and subfolders
        paths = []
        for cl in range(num_classes):
            class_directory = f"class-{cl}"
            classpaths = [
                class_directory,
                os.path.join(class_directory, "subfolder-1"),
                os.path.join(class_directory, "subfolder-2"),
                os.path.join(class_directory, "subfolder-1", "sub-subfolder"),
            ]
            for path in classpaths:
                os.mkdir(os.path.join(tmpdir, path))
            paths.append(classpaths)

        # save the images in the paths
        count = 0
        filenames = []
        for test_images in all_test_images:
            for im in test_images:
                # rotate image class
                im_class = count % num_classes
                # rotate subfolders
                classpaths = paths[im_class]
                filename = os.path.join(
                    classpaths[count % len(classpaths)],
                    f"image-{count}.png",
                )
                filenames.append(filename)
                im.save(os.path.join(tmpdir.full_path, filename))
                count += 1

        # create dataframe
        classes = np.random.randint(num_classes, size=len(filenames))
        classes = [str(c) for c in classes]
        df = pd.DataFrame({"filename": filenames, "class": classes})

        # create iterator
        generator = image.ImageDataGenerator()
        df_iterator = generator.flow_from_dataframe(
            df, tmpdir.full_path, class_mode="binary"
        )

        # Test the number of items in iterator
        self.assertLen(filenames, df_iterator.n)
        self.assertEqual(set(df_iterator.filenames), set(filenames))

    def test_dataframe_iterator_classes_indices_order(self):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(include_rgba=True)
        # save the images in the paths
        count = 0
        filenames = []
        for test_images in all_test_images:
            for im in test_images:
                filename = f"image-{count}.png"
                im.save(os.path.join(tmpdir.full_path, filename))
                filenames.append(filename)
                count += 1

        # Test the class_indices without classes input
        generator = image.ImageDataGenerator()
        label_opt = ["a", "b", ["a"], ["b"], ["a", "b"], ["b", "a"]]
        df_f = pd.DataFrame(
            {
                "filename": filenames,
                "class": ["a", "b"]
                + [random.choice(label_opt) for _ in filenames[:-2]],
            }
        )
        flow_forward_iter = generator.flow_from_dataframe(
            df_f, tmpdir.full_path
        )
        label_rev = ["b", "a", ["b"], ["a"], ["b", "a"], ["a", "b"]]
        df_r = pd.DataFrame(
            {
                "filename": filenames,
                "class": ["b", "a"]
                + [random.choice(label_rev) for _ in filenames[:-2]],
            }
        )
        flow_backward_iter = generator.flow_from_dataframe(
            df_r, tmpdir.full_path
        )

        # check class_indices
        self.assertEqual(
            flow_forward_iter.class_indices, flow_backward_iter.class_indices
        )

        # Test the class_indices with classes input
        generator_2 = image.ImageDataGenerator()
        df_f2 = pd.DataFrame(
            [["data/A.jpg", "A"], ["data/B.jpg", "B"]],
            columns=["filename", "class"],
        )
        flow_forward = generator_2.flow_from_dataframe(
            df_f2, classes=["A", "B"]
        )
        df_b2 = pd.DataFrame(
            [["data/A.jpg", "A"], ["data/B.jpg", "B"]],
            columns=["filename", "class"],
        )
        flow_backward = generator_2.flow_from_dataframe(
            df_b2, classes=["B", "A"]
        )

        # check class_indices
        self.assertNotEqual(
            flow_forward.class_indices, flow_backward.class_indices
        )


@test_utils.run_v2_only
class TestImageDataGenerator(test_combinations.TestCase):
    def test_image_data_generator(self):
        all_test_images = _generate_test_images(include_rgba=True)
        for test_images in all_test_images:
            img_list = []
            for im in test_images:
                img_list.append(image_utils.img_to_array(im)[None, ...])

            image.ImageDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=0.0,
                brightness_range=(1, 5),
                fill_mode="nearest",
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                interpolation_order=1,
            )

    def test_image_data_generator_with_validation_split(self):
        all_test_images = _generate_test_images(include_rgba=True)
        for test_images in all_test_images:
            img_list = []
            for im in test_images:
                img_list.append(image_utils.img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            labels = np.concatenate(
                [
                    np.zeros((int(len(images) / 2),)),
                    np.ones((int(len(images) / 2),)),
                ]
            )
            generator = image.ImageDataGenerator(validation_split=0.5)

            # training and validation sets would have different
            # number of classes, because labels are sorted
            with self.assertRaisesRegex(
                ValueError,
                "Training and validation subsets have "
                "different number of classes",
            ):
                generator.flow(
                    images,
                    labels,
                    shuffle=False,
                    batch_size=10,
                    subset="validation",
                )

            # test non categorical labels with validation split
            generator.flow(
                images,
                labels,
                shuffle=False,
                batch_size=10,
                ignore_class_split=True,
                subset="validation",
            )

            labels = np.concatenate(
                [
                    np.zeros((int(len(images) / 4),)),
                    np.ones((int(len(images) / 4),)),
                    np.zeros((int(len(images) / 4),)),
                    np.ones((int(len(images) / 4),)),
                ]
            )

            seq = generator.flow(
                images,
                labels,
                shuffle=False,
                batch_size=10,
                subset="validation",
            )

            _, y = seq[0]
            self.assertLen(np.unique(y), 2)

            seq = generator.flow(
                images, labels, shuffle=False, batch_size=10, subset="training"
            )
            _, y2 = seq[0]
            self.assertLen(np.unique(y2), 2)

            with self.assertRaises(ValueError):
                generator.flow(
                    images,
                    np.arange(images.shape[0]),
                    shuffle=False,
                    batch_size=3,
                    subset="foo",
                )

    def test_image_data_generator_with_split_value_error(self):
        with self.assertRaises(ValueError):
            image.ImageDataGenerator(validation_split=5)

    def test_image_data_generator_invalid_data(self):
        generator = image.ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format="channels_last",
        )
        # Test fit with invalid data
        with self.assertRaises(ValueError):
            x = np.random.random((3, 10, 10))
            generator.fit(x)

        # Test flow with invalid data
        with self.assertRaises(ValueError):
            x = np.random.random((32, 10, 10))
            generator.flow(np.arange(x.shape[0]))

    def test_image_data_generator_fit(self):
        generator = image.ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            rotation_range=90.0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.5,
            zoom_range=(0.2, 0.2),
            channel_shift_range=0.0,
            brightness_range=(1, 5),
            fill_mode="nearest",
            cval=0.5,
            horizontal_flip=True,
            vertical_flip=True,
            interpolation_order=1,
            data_format="channels_last",
        )
        x = np.random.random((32, 10, 10, 3))
        generator.fit(x, augment=True)
        # Test grayscale
        x = np.random.random((32, 10, 10, 1))
        generator.fit(x)
        # Test RBG
        x = np.random.random((32, 10, 10, 3))
        generator.fit(x)
        # Test more samples than dims
        x = np.random.random((32, 4, 4, 1))
        generator.fit(x)
        generator = image.ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            rotation_range=90.0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.5,
            zoom_range=(0.2, 0.2),
            channel_shift_range=0.0,
            brightness_range=(1, 5),
            fill_mode="nearest",
            cval=0.5,
            horizontal_flip=True,
            vertical_flip=True,
            interpolation_order=1,
            data_format="channels_first",
        )
        x = np.random.random((32, 10, 10, 3))
        generator.fit(x, augment=True)
        # Test grayscale
        x = np.random.random((32, 1, 10, 10))
        generator.fit(x)
        # Test RBG
        x = np.random.random((32, 3, 10, 10))
        generator.fit(x)
        # Test more samples than dims
        x = np.random.random((32, 1, 4, 4))
        generator.fit(x)

    def test_image_data_generator_flow(self):
        tmpdir = self.create_tempdir()
        all_test_images = _generate_test_images(include_rgba=True)
        for test_images in all_test_images:
            img_list = []
            for im in test_images:
                img_list.append(image_utils.img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            dsize = images.shape[0]
            generator = image.ImageDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=0.0,
                brightness_range=(1, 5),
                fill_mode="nearest",
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                interpolation_order=1,
            )

            generator.flow(
                images,
                np.arange(images.shape[0]),
                shuffle=False,
                save_to_dir=tmpdir.full_path,
                batch_size=3,
            )

            generator.flow(
                images,
                np.arange(images.shape[0]),
                shuffle=False,
                sample_weight=np.arange(images.shape[0]) + 1,
                save_to_dir=tmpdir.full_path,
                batch_size=3,
            )

            # Test with `shuffle=True`
            generator.flow(
                images,
                np.arange(images.shape[0]),
                shuffle=True,
                save_to_dir=tmpdir.full_path,
                batch_size=3,
                seed=42,
            )

            # Test without y
            generator.flow(
                images,
                None,
                shuffle=True,
                save_to_dir=tmpdir.full_path,
                batch_size=3,
            )

            # Test with a single miscellaneous input data array
            x_misc1 = np.random.random(dsize)
            generator.flow(
                (images, x_misc1), np.arange(dsize), shuffle=False, batch_size=2
            )

            # Test with two miscellaneous inputs
            x_misc2 = np.random.random((dsize, 3, 3))
            generator.flow(
                (images, [x_misc1, x_misc2]),
                np.arange(dsize),
                shuffle=False,
                batch_size=2,
            )

            # Test cases with `y = None`
            generator.flow(images, None, batch_size=3)
            generator.flow((images, x_misc1), None, batch_size=3, shuffle=False)
            generator.flow(
                (images, [x_misc1, x_misc2]), None, batch_size=3, shuffle=False
            )
            generator = image.ImageDataGenerator(validation_split=0.2)
            generator.flow(images, batch_size=3)

            # Test some failure cases:
            x_misc_err = np.random.random((dsize + 1, 3, 3))
            with self.assertRaisesRegex(ValueError, "All of the arrays in"):
                generator.flow(
                    (images, x_misc_err), np.arange(dsize), batch_size=3
                )

            with self.assertRaisesRegex(
                ValueError, r"`x` \(images tensor\) and `y` \(labels\)"
            ):
                generator.flow(
                    (images, x_misc1), np.arange(dsize + 1), batch_size=3
                )

            # Test `flow` behavior as Sequence
            generator.flow(
                images,
                np.arange(images.shape[0]),
                shuffle=False,
                save_to_dir=tmpdir.full_path,
                batch_size=3,
            )

            # Test with `shuffle=True`
            generator.flow(
                images,
                np.arange(images.shape[0]),
                shuffle=True,
                save_to_dir=tmpdir.full_path,
                batch_size=3,
                seed=123,
            )

        # test order_interpolation
        labels = np.array(
            [
                [2, 2, 0, 2, 2],
                [1, 3, 2, 3, 1],
                [2, 1, 0, 1, 2],
                [3, 1, 0, 2, 0],
                [3, 1, 3, 2, 1],
            ]
        )

        label_generator = image.ImageDataGenerator(
            rotation_range=90.0, interpolation_order=0
        )
        label_generator.flow(x=labels[np.newaxis, ..., np.newaxis], seed=123)

    def test_valid_args(self):
        with self.assertRaises(ValueError):
            image.ImageDataGenerator(brightness_range=0.1)

    def test_batch_standardize(self):
        all_test_images = _generate_test_images(include_rgba=True)
        # ImageDataGenerator.standardize should work on batches
        for test_images in all_test_images:
            img_list = []
            for im in test_images:
                img_list.append(image_utils.img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image.ImageDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=0.0,
                brightness_range=(1, 5),
                fill_mode="nearest",
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
            )
            generator.fit(images, augment=True)

            transformed = np.copy(images)
            for i, im in enumerate(transformed):
                transformed[i] = generator.random_transform(im)
            transformed = generator.standardize(transformed)

    def test_deterministic_transform(self):
        x = np.ones((32, 32, 3))
        generator = image.ImageDataGenerator(
            rotation_range=90, fill_mode="constant"
        )
        x = np.random.random((32, 32, 3))
        self.assertAllClose(
            generator.apply_transform(x, {"flip_vertical": True}), x[::-1, :, :]
        )
        self.assertAllClose(
            generator.apply_transform(x, {"flip_horizontal": True}),
            x[:, ::-1, :],
        )
        x = np.ones((3, 3, 3))
        x_rotated = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            ]
        )
        self.assertAllClose(
            generator.apply_transform(x, {"theta": 45}), x_rotated
        )

    def test_random_transforms(self):
        x = np.random.random((2, 28, 28))
        # Test get_random_transform with predefined seed
        seed = 1
        generator = image.ImageDataGenerator(
            rotation_range=90.0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.5,
            zoom_range=0.2,
            channel_shift_range=0.1,
            brightness_range=(1, 5),
            horizontal_flip=True,
            vertical_flip=True,
        )
        transform_dict = generator.get_random_transform(x.shape, seed)
        transform_dict2 = generator.get_random_transform(x.shape, seed * 2)
        self.assertNotEqual(transform_dict["theta"], 0)
        self.assertNotEqual(transform_dict["theta"], transform_dict2["theta"])
        self.assertNotEqual(transform_dict["tx"], 0)
        self.assertNotEqual(transform_dict["tx"], transform_dict2["tx"])
        self.assertNotEqual(transform_dict["ty"], 0)
        self.assertNotEqual(transform_dict["ty"], transform_dict2["ty"])
        self.assertNotEqual(transform_dict["shear"], 0)
        self.assertNotEqual(transform_dict["shear"], transform_dict2["shear"])
        self.assertNotEqual(transform_dict["zx"], 0)
        self.assertNotEqual(transform_dict["zx"], transform_dict2["zx"])
        self.assertNotEqual(transform_dict["zy"], 0)
        self.assertNotEqual(transform_dict["zy"], transform_dict2["zy"])
        self.assertNotEqual(transform_dict["channel_shift_intensity"], 0)
        self.assertNotEqual(
            transform_dict["channel_shift_intensity"],
            transform_dict2["channel_shift_intensity"],
        )
        self.assertNotEqual(transform_dict["brightness"], 0)
        self.assertNotEqual(
            transform_dict["brightness"], transform_dict2["brightness"]
        )

        # Test get_random_transform without any randomness
        generator = image.ImageDataGenerator()
        transform_dict = generator.get_random_transform(x.shape, seed)
        self.assertEqual(transform_dict["theta"], 0)
        self.assertEqual(transform_dict["tx"], 0)
        self.assertEqual(transform_dict["ty"], 0)
        self.assertEqual(transform_dict["shear"], 0)
        self.assertEqual(transform_dict["zx"], 1)
        self.assertEqual(transform_dict["zy"], 1)
        self.assertIsNone(transform_dict["channel_shift_intensity"], None)
        self.assertIsNone(transform_dict["brightness"], None)

    def test_fit_rescale(self):
        all_test_images = _generate_test_images(include_rgba=True)
        rescale = 1.0 / 255

        for test_images in all_test_images:
            img_list = []
            for im in test_images:
                img_list.append(image_utils.img_to_array(im)[None, ...])
            images = np.vstack(img_list)

            # featurewise_center test
            generator = image.ImageDataGenerator(
                rescale=rescale, featurewise_center=True, dtype="float64"
            )
            generator.fit(images)
            batch = generator.flow(images, batch_size=8).next()
            self.assertLess(abs(np.mean(batch)), 1e-6)

            # featurewise_std_normalization test
            generator = image.ImageDataGenerator(
                rescale=rescale,
                featurewise_center=True,
                featurewise_std_normalization=True,
                dtype="float64",
            )
            generator.fit(images)
            batch = generator.flow(images, batch_size=8).next()
            self.assertLess(abs(np.mean(batch)), 1e-6)
            self.assertLess(abs(1 - np.std(batch)), 1e-5)

            # zca_whitening test
            generator = image.ImageDataGenerator(
                rescale=rescale,
                featurewise_center=True,
                zca_whitening=True,
                dtype="float64",
            )
            generator.fit(images)
            batch = generator.flow(images, batch_size=8).next()
            batch = np.reshape(
                batch,
                (
                    batch.shape[0],
                    batch.shape[1] * batch.shape[2] * batch.shape[3],
                ),
            )
            # Y * Y_T = n * I, where Y = W * X
            identity = np.dot(batch, batch.T) / batch.shape[0]
            self.assertTrue(
                (
                    (np.abs(identity) - np.identity(identity.shape[0])) < 1e-6
                ).all()
            )


@test_utils.run_v2_only
class TestAffineTransformations(test_combinations.TestCase):
    def test_random_transforms(self):
        x = np.random.random((2, 28, 28))
        self.assertEqual(image.random_rotation(x, 45).shape, (2, 28, 28))
        self.assertEqual(image.random_shift(x, 1, 1).shape, (2, 28, 28))
        self.assertEqual(image.random_shear(x, 20).shape, (2, 28, 28))
        self.assertEqual(image.random_channel_shift(x, 20).shape, (2, 28, 28))

    def test_deterministic_transform(self):
        x = np.ones((3, 3, 3))
        x_rotated = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            ]
        )
        self.assertAllClose(
            image.apply_affine_transform(
                x,
                theta=45,
                row_axis=0,
                col_axis=1,
                channel_axis=2,
                fill_mode="constant",
            ),
            x_rotated,
        )

    def test_matrix_center(self):
        x = np.expand_dims(
            np.array(
                [
                    [0, 1],
                    [0, 0],
                ]
            ),
            -1,
        )
        x_rotated90 = np.expand_dims(
            np.array(
                [
                    [1, 0],
                    [0, 0],
                ]
            ),
            -1,
        )

        self.assertAllClose(
            image.apply_affine_transform(
                x, theta=90, row_axis=0, col_axis=1, channel_axis=2
            ),
            x_rotated90,
        )

    def test_translation(self):
        x = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        x_up = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        x_dn = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )
        x_left = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        x_right = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ]
        )

        # Channels first
        x_test = np.expand_dims(x, 0)

        # Horizontal translation
        self.assertAllEqual(
            x_left, np.squeeze(image.apply_affine_transform(x_test, tx=1))
        )
        self.assertAllEqual(
            x_right, np.squeeze(image.apply_affine_transform(x_test, tx=-1))
        )

        # change axes: x<->y
        self.assertAllEqual(
            x_left,
            np.squeeze(
                image.apply_affine_transform(
                    x_test, ty=1, row_axis=2, col_axis=1
                )
            ),
        )
        self.assertAllEqual(
            x_right,
            np.squeeze(
                image.apply_affine_transform(
                    x_test, ty=-1, row_axis=2, col_axis=1
                )
            ),
        )

        # Vertical translation
        self.assertAllEqual(
            x_up, np.squeeze(image.apply_affine_transform(x_test, ty=1))
        )
        self.assertAllEqual(
            x_dn, np.squeeze(image.apply_affine_transform(x_test, ty=-1))
        )

        # change axes: x<->y
        self.assertAllEqual(
            x_up,
            np.squeeze(
                image.apply_affine_transform(
                    x_test, tx=1, row_axis=2, col_axis=1
                )
            ),
        )
        self.assertAllEqual(
            x_dn,
            np.squeeze(
                image.apply_affine_transform(
                    x_test, tx=-1, row_axis=2, col_axis=1
                )
            ),
        )

        # Channels last
        x_test = np.expand_dims(x, -1)

        # Horizontal translation
        self.assertAllEqual(
            x_left,
            np.squeeze(
                image.apply_affine_transform(
                    x_test, tx=1, row_axis=0, col_axis=1, channel_axis=2
                )
            ),
        )
        self.assertAllEqual(
            x_right,
            np.squeeze(
                image.apply_affine_transform(
                    x_test, tx=-1, row_axis=0, col_axis=1, channel_axis=2
                )
            ),
        )

        # change axes: x<->y
        self.assertAllEqual(
            x_left,
            np.squeeze(
                image.apply_affine_transform(
                    x_test, ty=1, row_axis=1, col_axis=0, channel_axis=2
                )
            ),
        )
        self.assertAllEqual(
            x_right,
            np.squeeze(
                image.apply_affine_transform(
                    x_test, ty=-1, row_axis=1, col_axis=0, channel_axis=2
                )
            ),
        )

        # Vertical translation
        self.assertAllEqual(
            x_up,
            np.squeeze(
                image.apply_affine_transform(
                    x_test, ty=1, row_axis=0, col_axis=1, channel_axis=2
                )
            ),
        )
        self.assertAllEqual(
            x_dn,
            np.squeeze(
                image.apply_affine_transform(
                    x_test, ty=-1, row_axis=0, col_axis=1, channel_axis=2
                )
            ),
        )

        # change axes: x<->y
        self.assertAllEqual(
            x_up,
            np.squeeze(
                image.apply_affine_transform(
                    x_test, tx=1, row_axis=1, col_axis=0, channel_axis=2
                )
            ),
        )
        self.assertAllEqual(
            x_dn,
            np.squeeze(
                image.apply_affine_transform(
                    x_test, tx=-1, row_axis=1, col_axis=0, channel_axis=2
                )
            ),
        )

    def test_random_zoom(self):
        x = np.random.random((2, 28, 28))
        self.assertEqual(image.random_zoom(x, (5, 5)).shape, (2, 28, 28))
        self.assertAllClose(x, image.random_zoom(x, (1, 1)))

    def test_random_zoom_error(self):
        with self.assertRaises(ValueError):
            image.random_zoom(0, zoom_range=[0])

    def test_random_brightness_error(self):
        with self.assertRaises(ValueError):
            image.random_brightness(0, [0])

    def test_random_brightness_scale(self):
        img = np.ones((1, 1, 3)) * 128
        zeros = np.zeros((1, 1, 3))
        must_be_128 = image.random_brightness(img, [1, 1], False)
        self.assertAllEqual(img, must_be_128)
        must_be_0 = image.random_brightness(img, [1, 1], True)
        self.assertAllEqual(zeros, must_be_0)

    def test_random_brightness_scale_outside_range_positive(self):
        img = np.ones((1, 1, 3)) * 1024
        zeros = np.zeros((1, 1, 3))
        must_be_1024 = image.random_brightness(img, [1, 1], False)
        self.assertAllEqual(img, must_be_1024)
        must_be_0 = image.random_brightness(img, [1, 1], True)
        self.assertAllEqual(zeros, must_be_0)

    def test_random_brightness_scale_outside_range_negative(self):
        img = np.ones((1, 1, 3)) * -1024
        zeros = np.zeros((1, 1, 3))
        must_be_neg_1024 = image.random_brightness(img, [1, 1], False)
        self.assertAllEqual(img, must_be_neg_1024)
        must_be_0 = image.random_brightness(img, [1, 1], True)
        self.assertAllEqual(zeros, must_be_0)


if __name__ == "__main__":
    tf.test.main()
