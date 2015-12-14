
## ImageDataGenerator

```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    horizontal_flip=False,
    vertical_flip=False)
```

Generate batches of tensor image data with real-time data augmentation.

- __Arguments__:
    - __featurewise_center__: Boolean. Set input mean to 0 over the dataset.
    - __samplewise_center__: Boolean. Set each sample mean to 0.
    - __featurewise_std_normalization__: Boolean. Divide inputs by std of the dataset.
    - __samplewise_std_normalization__: Boolean. Divide each input by its std.
    - __zca_whitening__: Boolean. Apply ZCA whitening.
    - __rotation_range__: Int. Degree range for random rotations.
    - __width_shift_range__: Float (fraction of total width). Range for random horizontal shifts.
    - __height_shift_range__: Float (fraction of total height). Range for random vertical shifts.
    - __horizontal_flip__: Boolean. Randomly flip inputs horizontally.
    - __vertical_flip__: Boolean. Randomly flip inputs vertically.

- __Methods__:
    - __fit(X)__: Required if featurewise_center or featurewise_std_normalization or zca_whitening. Compute necessary quantities on some sample data.
        - __Arguments__: 
            - __X__: sample data.
            - __augment__: Boolean (default: False). Whether to fit on randomly augmented samples.
            - __rounds__: int (default: 1). If augment, how many augmentation passes over the data to use.
    - __flow(X, y)__:
        - __Arguments__: 
            - __X__: data.
            - __y__: labels.
            - __batch_size__: int (default: 32).
            - __shuffle__: boolean (defaut: False).
            - __save_to_dir__: None or str. This allows you to optimally specify a directory to which to save the augmented pictures being generated (useful for visualizing what you are doing). 
            - __save_prefix__: str. Prefix to use for filenames of saved pictures.
            - __save_format__: one of "png", jpeg". 

- __Example__:
```python
(X_train, y_train), (X_test, y_test) = cifar10.load_data(test_split=0.1)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization 
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)

for e in range(nb_epoch):
    print 'Epoch', e
    # batch train with realtime data augmentation
    for X_batch, Y_batch in datagen.flow(X_train, Y_train):
        loss = model.train(X_batch, Y_batch)
```