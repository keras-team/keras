This is intended to serve as a conversion module for Caffe models to Keras.

In order to convert a model you just need the .caffemodel weights and the .prototxt deploy or train file. In any case you will need to include the input image dimensions as a header to the .prototxt network structure as if it was a deploy model.

The file caffe2keras.py can be used as a command line interface for converting any model.



This code is a modified and improvied version by Marc Bolaños of the code originally written by Pranav Shyam.

Contact email: marc.bolanos@ub.edu
GitHub page: https://github.com/MarcBS
