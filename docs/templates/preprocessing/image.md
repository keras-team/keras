
## ImageDataGenerator

```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=K.image_data_format())
```

Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches) indefinitely.

- __Arguments__:
    - __featurewise_center__: Boolean. Set input mean to 0 over the dataset, feature-wise.
    - __samplewise_center__: Boolean. Set each sample mean to 0.
    - __featurewise_std_normalization__: Boolean. Divide inputs by std of the dataset, feature-wise.
    - __samplewise_std_normalization__: Boolean. Divide each input by its std.
    - __zca_epsilon__: epsilon for ZCA whitening. Default is 1e-6.
    - __zca_whitening__: Boolean. Apply ZCA whitening.
    - __rotation_range__: Int. Degree range for random rotations.
    - __width_shift_range__: Float (fraction of total width). Range for random horizontal shifts.
    - __height_shift_range__: Float (fraction of total height). Range for random vertical shifts.
    - __shear_range__: Float. Shear Intensity (Shear angle in counter-clockwise direction as radians)
    - __zoom_range__: Float or [lower, upper]. Range for random zoom. If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
    - __channel_shift_range__: Float. Range for random channel shifts.
    - __fill_mode__: One of {"constant", "nearest", "reflect" or "wrap"}.  Points outside the boundaries of the input are filled according to the given mode.
    - __cval__: Float or Int. Value used for points outside the boundaries when `fill_mode = "constant"`.
    - __horizontal_flip__: Boolean. Randomly flip inputs horizontally.
    - __vertical_flip__: Boolean. Randomly flip inputs vertically.
    - __rescale__: rescaling factor. Defaults to None. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation).
    - __preprocessing_function__: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
    - _data_format_: One of {"channels_first", "channels_last"}.
        "channels_last" mode means that the images should have shape `(samples, height, width, channels)`,
        "channels_first" mode means that the images should have shape `(samples, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

- __Methods__:
    - __fit(x)__: Compute the internal data stats related to the data-dependent transformations, based on an array of sample data.
        Only required if featurewise_center or featurewise_std_normalization or zca_whitening.
        - __Arguments__:
            - __x__: sample data. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            - __augment__: Boolean (default: False). Whether to fit on randomly augmented samples.
            - __rounds__: int (default: 1). If augment, how many augmentation passes over the data to use.
            - __seed__: int (default: None). Random seed.
    - __flow(x, y)__: Takes numpy data & label arrays, and generates batches of augmented/normalized data. Yields batches indefinitely, in an infinite loop.
        - __Arguments__:
            - __x__: data. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            - __y__: labels.
            - __batch_size__: int (default: 32).
            - __shuffle__: boolean (default: True).
            - __seed__: int (default: None).
            - __save_to_dir__: None or str (default: None). This allows you to optimally specify a directory to which to save the augmented pictures being generated (useful for visualizing what you are doing).
            - __save_prefix__: str (default: `''`). Prefix to use for filenames of saved pictures (only relevant if `save_to_dir` is set).
            - __save_format__: one of "png", "jpeg" (only relevant if `save_to_dir` is set). Default: "png".
        - __yields__: Tuples of `(x, y)` where `x` is a numpy array of image data and `y` is a numpy array of corresponding labels.
            The generator loops indefinitely.
    - __flow_from_directory(directory)__: Takes the path to a directory, and generates batches of augmented/normalized data. Yields batches indefinitely, in an infinite loop.
        - __Arguments__:
            - __directory__: path to the target directory. It should contain one subdirectory per class.
                Any PNG, JPG or BMP images inside each of the subdirectories directory tree will be included in the generator.
                See [this script](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d) for more details.
            - __target_size__: tuple of integers `(height, width)`, default: `(256, 256)`. 
                The dimensions to which all images found will be resized.
            - __color_mode__: one of "grayscale", "rbg". Default: "rgb". Whether the images will be converted to have 1 or 3 color channels.
            - __classes__: optional list of class subdirectories (e.g. `['dogs', 'cats']`). Default: None. If not provided, the list of classes will be automatically inferred from the subdirectory names/structure under `directory`, where each subdirectory will be treated as a different class (and the order of the classes, which will map to the label indices, will be alphanumeric). The dictionary containing the mapping from class names to class indices can be obtained via the attribute `class_indices`.
            - __class_mode__: one of "categorical", "binary", "sparse" or None. Default: "categorical". Determines the type of label arrays that are returned: "categorical" will be 2D one-hot encoded labels, "binary" will be 1D binary labels, "sparse" will be 1D integer labels. If None, no labels are returned (the generator will only yield batches of image data, which is useful to use `model.predict_generator()`, `model.evaluate_generator()`, etc.). Please note that in case of class_mode None, the data still needs to reside in a subdirectory of `directory` for it to work correctly.
            - __batch_size__: size of the batches of data (default: 32).
            - __shuffle__: whether to shuffle the data (default: True)
            - __seed__: optional random seed for shuffling and transformations.
            - __save_to_dir__: None or str (default: None). This allows you to optimally specify a directory to which to save the augmented pictures being generated (useful for visualizing what you are doing).
            - __save_prefix__: str. Prefix to use for filenames of saved pictures (only relevant if `save_to_dir` is set).
            - __save_format__: one of "png", "jpeg" (only relevant if `save_to_dir` is set). Default: "png".
            - __follow_links__: whether to follow symlinks inside class subdirectories (default: False).


- __Examples__:

Example of using `.flow(x, y)`:

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# here's a more "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
```

Example of using `.flow_from_directory(directory)`:

```python
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```

Example of transforming images and masks together.

```python
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
```
