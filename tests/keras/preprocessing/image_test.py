import pytest
from keras.preprocessing import image
from PIL import Image
import numpy as np
import os


class TestImage:

    def setup_class(cls):
        img_w = img_h = 20
        rgb_images = []
        gray_images = []
        for n in range(8):
            bias = np.random.rand(img_w, img_h, 1) * 64
            variance = np.random.rand(img_w, img_h, 1) * (255 - 64)
            imarray = np.random.rand(img_w, img_h, 3) * variance + bias
            im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
            rgb_images.append(im)

            imarray = np.random.rand(img_w, img_h, 1) * variance + bias
            im = Image.fromarray(imarray.astype('uint8').squeeze()).convert('L')
            gray_images.append(im)

        cls.all_test_images = [rgb_images, gray_images]

    def teardown_class(cls):
        del cls.all_test_images

    def test_image_data_generator(self, tmpdir):
        for test_images in self.all_test_images:
            img_list = []
            for im in test_images:
                img_list.append(image.img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image.ImageDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=0.,
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True)
            generator.fit(images, augment=True)

            for x, y in generator.flow(images, np.arange(images.shape[0]),
                                       shuffle=True, save_to_dir=str(tmpdir)):
                assert x.shape[1:] == images.shape[1:]
                break

    def test_image_data_generator_invalid_data(self):
        generator = image.ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_last')
        # Test fit with invalid data
        with pytest.raises(ValueError):
            x = np.random.random((3, 10, 10))
            generator.fit(x)

        # Test flow with invalid data
        with pytest.raises(ValueError):
            x = np.random.random((32, 10, 10))
            generator.flow(np.arange(x.shape[0]))

    def test_image_data_generator_fit(self):
        generator = image.ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            zoom_range=(0.2, 0.2),
            data_format='channels_last')
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
            data_format='channels_first')
        # Test grayscale
        x = np.random.random((32, 1, 10, 10))
        generator.fit(x)
        # Test RBG
        x = np.random.random((32, 3, 10, 10))
        generator.fit(x)

    def test_directory_iterator(self, tmpdir):
        num_classes = 2

        # create folders and subfolders
        paths = []
        for cl in range(num_classes):
            class_directory = 'class-{}'.format(cl)
            classpaths = [
                class_directory,
                os.path.join(class_directory, 'subfolder-1'),
                os.path.join(class_directory, 'subfolder-2'),
                os.path.join(class_directory, 'subfolder-1', 'sub-subfolder')
            ]
            for path in classpaths:
                tmpdir.join(path).mkdir()
            paths.append(classpaths)

        # save the images in the paths
        count = 0
        filenames = []
        for test_images in self.all_test_images:
            for im in test_images:
                # rotate image class
                im_class = count % num_classes
                # rotate subfolders
                classpaths = paths[im_class]
                filename = os.path.join(classpaths[count % len(classpaths)], 'image-{}.jpg'.format(count))
                filenames.append(filename)
                im.save(str(tmpdir / filename))
                count += 1

        # create iterator
        generator = image.ImageDataGenerator()
        dir_iterator = generator.flow_from_directory(str(tmpdir))

        # check number of classes and images
        assert(len(dir_iterator.class_indices) == num_classes)
        assert(len(dir_iterator.classes) == count)
        assert(sorted(dir_iterator.filenames) == sorted(filenames))

        # Test invalid use cases
        with pytest.raises(ValueError):
            generator.flow_from_directory(str(tmpdir), color_mode='cmyk')
        with pytest.raises(ValueError):
            generator.flow_from_directory(str(tmpdir), class_mode='output')

    def test_directory_iterator_class_mode_input(self, tmpdir):
        tmpdir.join('class-1').mkdir()

        # save the images in the paths
        count = 0
        for test_images in self.all_test_images:
            for im in test_images:
                filename = str(tmpdir / 'class-1' / 'image-{}.jpg'.format(count))
                im.save(filename)
                count += 1

        # create iterator
        generator = image.ImageDataGenerator()
        dir_iterator = generator.flow_from_directory(str(tmpdir), class_mode='input')
        batch = next(dir_iterator)

        # check if input and output have the same shape
        assert(batch[0].shape == batch[1].shape)
        # check if the input and output images are not the same numpy array
        input_img = batch[0][0]
        output_img = batch[1][0]
        output_img[0][0][0] += 1
        assert(input_img[0][0][0] != output_img[0][0][0])

    def test_img_utils(self):
        height, width = 10, 8

        # Test th data format
        x = np.random.random((3, height, width))
        img = image.array_to_img(x, data_format='channels_first')
        assert img.size == (width, height)
        x = image.img_to_array(img, data_format='channels_first')
        assert x.shape == (3, height, width)
        # Test 2D
        x = np.random.random((1, height, width))
        img = image.array_to_img(x, data_format='channels_first')
        assert img.size == (width, height)
        x = image.img_to_array(img, data_format='channels_first')
        assert x.shape == (1, height, width)

        # Test tf data format
        x = np.random.random((height, width, 3))
        img = image.array_to_img(x, data_format='channels_last')
        assert img.size == (width, height)
        x = image.img_to_array(img, data_format='channels_last')
        assert x.shape == (height, width, 3)
        # Test 2D
        x = np.random.random((height, width, 1))
        img = image.array_to_img(x, data_format='channels_last')
        assert img.size == (width, height)
        x = image.img_to_array(img, data_format='channels_last')
        assert x.shape == (height, width, 1)

        # Test invalid use case
        with pytest.raises(ValueError):
            x = np.random.random((height, width))  # not 3D
            img = image.array_to_img(x, data_format='channels_first')
        with pytest.raises(ValueError):
            x = np.random.random((height, width, 3))
            img = image.array_to_img(x, data_format='channels')  # unknown data_format
        with pytest.raises(ValueError):
            x = np.random.random((height, width, 5))  # neither RGB nor gray-scale
            img = image.array_to_img(x, data_format='channels_last')
        with pytest.raises(ValueError):
            x = np.random.random((height, width, 3))
            img = image.img_to_array(x, data_format='channels')  # unknown data_format
        with pytest.raises(ValueError):
            x = np.random.random((height, width, 5, 3))  # neither RGB nor gray-scale
            img = image.img_to_array(x, data_format='channels_last')

    def test_random_transforms(self):
        x = np.random.random((2, 28, 28))
        assert image.random_rotation(x, 45).shape == (2, 28, 28)
        assert image.random_shift(x, 1, 1).shape == (2, 28, 28)
        assert image.random_shear(x, 20).shape == (2, 28, 28)
        assert image.random_zoom(x, (5, 5)).shape == (2, 28, 28)
        assert image.random_channel_shift(x, 20).shape == (2, 28, 28)

    def test_batch_standardize(self):
        # ImageDataGenerator.standardize should work on batches
        for test_images in self.all_test_images:
            img_list = []
            for im in test_images:
                img_list.append(image.img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image.ImageDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=0.,
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True)
            generator.fit(images, augment=True)

            transformed = np.copy(images)
            for i, im in enumerate(transformed):
                transformed[i] = generator.random_transform(im)
            transformed = generator.standardize(transformed)


if __name__ == '__main__':
    pytest.main([__file__])
