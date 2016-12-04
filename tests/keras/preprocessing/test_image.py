import pytest
from keras.preprocessing.image import *
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
            variance = np.random.rand(img_w, img_h, 1) * (255-64)
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
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = ImageDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                channelwise=True, 
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

    def test_img_flip(self):
        x = np.array(range(4)).reshape([1, 1, 2, 2])
        assert (flip_axis(x, 0) == x).all()
        assert (flip_axis(x, 1) == x).all()
        assert (flip_axis(x, 2) == [[[[2, 3], [0, 1]]]]).all()
        assert (flip_axis(x, 3) == [[[[1, 0], [3, 2]]]]).all()

        dim_ordering_and_col_index = (('tf', 2), ('th', 3))
        for dim_ordering, col_index in dim_ordering_and_col_index:
            image_generator_th = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0,
                width_shift_range=0,
                height_shift_range=0,
                shear_range=0,
                zoom_range=0,
                channel_shift_range=0,
                horizontal_flip=True,
                vertical_flip=False,
                dim_ordering=dim_ordering).flow(x, [1])
            for i in range(10):
                potentially_flipped_x, _ = next(image_generator_th)
                assert ((potentially_flipped_x == x).all() or
                        (potentially_flipped_x == flip_axis(x, col_index)).all())

    def test_standardize(self): 
        eps = 1e-5

        for test_images in self.all_test_images:
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            imgs = np.vstack(img_list)
            for dim_ordering in ('tf', 'th'):
                if dim_ordering == 'th': 
                    imgs = imgs.swapaxes(1, 3)
                generator = ImageDataGenerator(
                    featurewise_center=True,
                    samplewise_center=False,
                    featurewise_std_normalization=True,
                    samplewise_std_normalization=False,
                    channelwise=False, 
                    zca_whitening=False,
                    rotation_range=0,
                    width_shift_range=0,
                    height_shift_range=0,
                    shear_range=0,
                    zoom_range=0,
                    channel_shift_range=0,
                    horizontal_flip=False,
                    vertical_flip=False,
                    dim_ordering=dim_ordering)
                generator.fit(imgs)

                eps = 1e-5

                standardized_imgs = np.array([generator.standardize(img) for img in imgs])
                assert np.all(np.abs(standardized_imgs.mean(axis=0)) < eps)
                assert np.all(np.abs(standardized_imgs.std(axis=0) - 1) < eps)

                generator.channelwise = True 
                generator.fit(imgs)
                collapse_axis = (0, generator.row_index, generator.col_index)

                standardized_imgs = np.array([generator.standardize(img) for img in imgs])
                assert np.all(np.abs(standardized_imgs.mean(axis=collapse_axis)) < eps)
                assert np.all(np.abs(standardized_imgs.std(axis=collapse_axis) - 1) < eps)

                generator.featurewise_center = False
                generator.featurewise_std_normalization = False
                generator.samplewise_center = True
                generator.samplewise_std_normalization = True
                generator.channelwise = False
                generator.fit(imgs)

                collapse_axis = generator.channel_index - 1
                standardized_img = generator.standardize(imgs[0])
                assert np.all(np.abs(standardized_img.mean(axis=collapse_axis)) < eps)
                assert np.all(np.abs(standardized_img.std(axis=collapse_axis) - 1) < eps)

                generator.channelwise = True 
                generator.fit(imgs)

                collapse_axis = (generator.row_index - 1, generator.col_index - 1)
                standardized_img = generator.standardize(imgs[0])
                assert np.all(np.abs(standardized_img.mean(axis=collapse_axis)) < eps)
                assert np.all(np.abs(standardized_img.std(axis=collapse_axis) - 1) < eps)

if __name__ == '__main__':
    pytest.main([__file__])
