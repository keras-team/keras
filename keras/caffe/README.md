This is intended to serve as a conversion module for Caffe models to Keras Functional API models.

In order to convert a model you just need the .caffemodel weights and the .prototxt deploy or train file. In any case you will need to include the input image dimensions as a header to the .prototxt network structure as if it was a deploy model (see an example [here](models)).

The file caffe2keras.py can be used as a command line interface for converting any model.
In the file [test_converted.py](test_converted.py) you can see an example on how to use a converted model.

This code is a modified and improved version by Marc Bolaños of the original pieces of code originally written by [Pranav Shyam](https://github.com/pranv) and [Antonella Cascitelli](https://github.com/lenlen).

Contact email: marc.bolanos@ub.edu
GitHub page: https://github.com/MarcBS
