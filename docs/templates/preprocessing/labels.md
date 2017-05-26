 
## to_categorical

```python
keras.utils.np_utils.to_categorical(labels, nb_classes=None)
```

Convert class vector (integers from 0 to nb_classes) to binary class matrix. For use with categorical_crossentropy.

- __Return__: numpy array of shape `(len(labels), nb_classes)`.

- __Arguments__:
    - __labels__: list of labels.
    - __nb_classes__: None or int. Maximum number of classes, this determines the number of columns in the returned matrix. Default: None, the number of columns defaults to the max integer in the labels array.
