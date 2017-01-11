import pytest
from keras.preprocessing import image
from PIL import Image
import numpy as np
import os
import shutil
import tempfile


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

    def test_image_data_generator(self):
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

            tmp_folder = tempfile.mkdtemp(prefix='test_images')
            for x, y in generator.flow(images, np.arange(images.shape[0]),
                                       shuffle=True, save_to_dir=tmp_folder):
                assert x.shape[1:] == images.shape[1:]
                break
            shutil.rmtree(tmp_folder)

    def test_image_data_generator_invalid_data(self):
        generator = image.ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            dim_ordering='tf')
        # Test fit with invalid data
        with pytest.raises(ValueError):
            x = np.random.random((3, 10, 10))
            generator.fit(x)
        with pytest.raises(ValueError):
            x = np.random.random((32, 3, 10, 10))
            generator.fit(x)
        with pytest.raises(ValueError):
            x = np.random.random((32, 10, 10, 5))
            generator.fit(x)
        # Test flow with invalid data
        with pytest.raises(ValueError):
            x = np.random.random((32, 10, 10, 5))
            generator.flow(np.arange(x.shape[0]))
        with pytest.raises(ValueError):
            x = np.random.random((32, 10, 10))
            generator.flow(np.arange(x.shape[0]))
        with pytest.raises(ValueError):
            x = np.random.random((32, 3, 10, 10))
            generator.flow(np.arange(x.shape[0]))

    def test_image_data_generator_fit(self):
        generator = image.ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            dim_ordering='tf')
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
            dim_ordering='th')
        # Test grayscale
        x = np.random.random((32, 1, 10, 10))
        generator.fit(x)
        # Test RBG
        x = np.random.random((32, 3, 10, 10))
        generator.fit(x)

    def test_directory_iterator(self):
        num_classes = 2
        tmp_folder = tempfile.mkdtemp(prefix='test_images')

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
                os.mkdir(os.path.join(tmp_folder, path))
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
                im.save(os.path.join(tmp_folder, filename))
                count += 1

        # create iterator
        generator = image.ImageDataGenerator()
        dir_iterator = generator.flow_from_directory(tmp_folder)

        # check number of classes and images
        assert(len(dir_iterator.class_indices) == num_classes)
        assert(len(dir_iterator.classes) == count)
        assert(sorted(dir_iterator.filenames) == sorted(filenames))
        shutil.rmtree(tmp_folder)

    def test_img_utils(self):
        height, width = 10, 8

        # Test th dim ordering
        x = np.random.random((3, height, width))
        img = image.array_to_img(x, dim_ordering='th')
        assert img.size == (width, height)
        x = image.img_to_array(img, dim_ordering='th')
        assert x.shape == (3, height, width)
        # Test 2D
        x = np.random.random((1, height, width))
        img = image.array_to_img(x, dim_ordering='th')
        assert img.size == (width, height)
        x = image.img_to_array(img, dim_ordering='th')
        assert x.shape == (1, height, width)

        # Test tf dim ordering
        x = np.random.random((height, width, 3))
        img = image.array_to_img(x, dim_ordering='tf')
        assert img.size == (width, height)
        x = image.img_to_array(img, dim_ordering='tf')
        assert x.shape == (height, width, 3)
        # Test 2D
        x = np.random.random((height, width, 1))
        img = image.array_to_img(x, dim_ordering='tf')
        assert img.size == (width, height)
        x = image.img_to_array(img, dim_ordering='tf')
        assert x.shape == (height, width, 1)


if __name__ == '__main__':
    pytest.main([__file__])
