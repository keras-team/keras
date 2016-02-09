This is intended to serve as a conversion module for Caffe models to Keras.

In order to convert a model you just need the .caffemodel weights and the .prototxt deploy or train file. In any case you will need to include the input image dimensions as a header to the .prototxt network structure as if it was a deploy model.

The file caffe2keras.py can be used as a command line interface for converting any model.
In the file test_converted.py we can see an example on how to use a converted model.

Although the models can be converted and used, it looks like for some lose in weights' precision the accuracy of the models are reduced. If anybody has an idea or a solution I will be grateful for their help.
As an example, after converting a model obtained after finetuning GoogleNet on Caffe, the test accuracy was reduced from 79% to 72%. Although, after performing a new and short finetuning on Keras, the accuracy was again improved to 77.5%.


This code is a modified and improved version by Marc Bolaños of the code originally written by Pranav Shyam.

Contact email: marc.bolanos@ub.edu
GitHub page: https://github.com/MarcBS
