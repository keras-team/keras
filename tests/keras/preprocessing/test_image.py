import pytest
from keras.preprocessing.image import *
from PIL import Image
import numpy as np
import os
import shutil


def setup_function(func):
    paths = ['test_images', 'test_images/rgb', 'test_images/gsc']
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

    img_w = img_h = 20
    for n in range(8):
        bias = np.random.rand(img_w, img_h, 1) * 64
        variance = np.random.rand(img_w, img_h, 1) * (255-64)
        imarray = np.random.rand(img_w, img_h, 3) * variance + bias
        im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        im.save('test_images/rgb/rgb_test_image_'+str(n)+'.png')

        imarray = np.random.rand(img_w, img_h, 1) * variance + bias
        im = Image.fromarray(imarray.astype('uint8').squeeze()).convert('L')
        im.save('test_images/gsc/gsc_test_image_'+str(n)+'.png')


def teardown_function(func):
    shutil.rmtree('test_images')


def test_image_data_generator():
    for color_mode in ['gsc', 'rgb']:
        file_list = list_pictures('test_images/' + color_mode)
        img_list = []
        for f in file_list:
            img_list.append(img_to_array(load_img(f))[None, ...])

        images = np.vstack(img_list)
        generator = ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            rotation_range=90.,
            width_shift_range=10.,
            height_shift_range=10.,
            shear_range=0.5,
            horizontal_flip=True,
            vertical_flip=True)
        generator.fit(images, augment=True)

        for x, y in generator.flow(images, np.arange(images.shape[0]),
                                   shuffle=True, save_to_dir='test_images'):
            assert x.shape[1:] == images.shape[1:]
            break


def test_img_flip():
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
            horizontal_flip=True,
            vertical_flip=False,
            dim_ordering=dim_ordering).flow(x, [1])
        for i in range(10):
            potentially_flipped_x, _ = next(image_generator_th)
            assert ((potentially_flipped_x == x).all() or
                    (potentially_flipped_x == flip_axis(x, col_index)).all())


if __name__ == '__main__':
    pytest.main([__file__])
