import pytest
from keras.preprocessing.image import *
from PIL import Image
import numpy as np
import os
import shutil

def setup_function(func):
    np.random.seed(1337)

    os.mkdir('test_images')
    os.mkdir('test_images/rgb')
    os.mkdir('test_images/gsc')

    img_w = img_h = 20
    for n in range(8):
        bias = np.random.rand(img_w,img_h,1)*64
        variance = np.random.rand(img_w,img_h,1)*(255-64)
        imarray = np.random.rand(img_w,img_h,3) * variance + bias
        im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        im.save('test_images/rgb/rgb_test_image_'+str(n)+'.png')

        imarray = np.random.rand(img_w,img_h,1) * variance + bias
        im = Image.fromarray(imarray.astype('uint8').squeeze()).convert('L')
        im.save('test_images/gsc/gsc_test_image_'+str(n)+'.png')

def teardown_function(func):
    shutil.rmtree('test_images')

def test_image_data_generator():
    np.random.seed(1337)

    for color_mode in ['gsc','rgb']:
        file_list = list_pictures('test_images/'+color_mode)
        img_list = []
        for f in file_list:
            img_list.append(img_to_array(load_img(f))[None,...])

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
            horizontal_flip=True,
            vertical_flip=True
        )

        generator.fit(images,augment=True)

        for x,y in generator.flow(images,np.arange(images.shape[0]), shuffle=True, save_to_dir='test_images'):
            assert x.shape[1:] == images.shape[1:]
            # TODO: make sure the normalization is working as inteded


if __name__ == '__main__':
    pytest.main([__file__])
