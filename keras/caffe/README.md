## Caffe to Keras converter

This is intended to serve as a conversion module for Caffe models to Keras Functional API models.

**Please, be aware that this feature is not regularly maintained.** Thus, some layers or parameter definitions introduced in newer versions of either Keras or Caffe might not be compatible with the converter.

**For this reason, any pull requests with updated versions of the caffe2keras converter are highly welcome!**


### Conversion

In order to convert a model you just need the .caffemodel weights and the .prototxt deploy or train file. In any case you will need to include the input image dimensions as a header to the .prototxt network structure as if it was a deploy model (see an example [here](models)) and also include an initial data layer:

```
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
}
```

Given the differences between Caffe and Keras when applying the MAX pooling opperation, in some occasions the MAX pooling layers must include a `pad: 1` value even if they did not include them in their original .prototxt.

The file caffe2keras.py can be used as a command line interface for converting any model the following way:

```
python caffe2keras.py -load_path 'models/' -prototxt 'train_val_for_keras.prototxt' -caffemodel 'bvlc_googlenet.caffemodel'
```

### Model usage

In the file [test_converted.py](test_converted.py) you can see an example on how to use a converted model.


### Acknowledgments

This code is a modified and improved version by Marc Bolaños of the original pieces of code originally written by [Pranav Shyam](https://github.com/pranv) and [Antonella Cascitelli](https://github.com/lenlen).

Contact email: marc.bolanos@ub.edu
GitHub page: https://github.com/MarcBS
