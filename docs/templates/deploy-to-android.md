# Deploying Keras Models In Android

## Introduction

Keras models need to be converted into [TensorFlow Lite](https://www.tensorflow.org/lite) model before
they are deployed in mobile devices. TensorFlow Lite provides a converter API and a command line tool
to carry out the conversion. The converter optimizes the model for better performance on memory and power constrained devices like mobile phones and embedded devices. TFLite's converter compresses the model in
order to reduce it's size and efficiency. The user should know that model compression sometimes hurts the
integrity of the model resulting in less accurate inference operations.

TFLite's [method of model conversion](https://medium.com/tensorflow/introducing-the-model-optimization-toolkit-for-tensorflow-254aca1ba0a3),
have reportedly shown significant reduction in model size and less drop in accuracy. Compressing and
pruning DNNs, in order make them small and more efficient have gained a lot of research interest over
the years, discussing them are beyond the scope of this tutorial. Rest of the tutorial focuses on converting the model using [TFLite](https://www.tensorflow.org/lite) and integrating it into an Android Application.

---
## Using the API

### Example (uses TF version `1.12`)
```python
>>> from tensorflow.contrib.lite import TFLiteConverter
>>> converter = TFLiteConverter.from_keras_model_file("keras_model.h5")
>>> tflite_model = converter.convert()
>>> open("converted_model.tflite", "wb").write(tflite_model)
```

Depending on the version of TensorFlow, the converter API resides on different corners of the framework.
Refer to the table below and run `help()` on the respective class to find ```.from_keras_model_file()``` method. This is the TFLite provided method to convert Keras models. The Keras model to be converted should be in `hdf5` format containing both the architecture and it's weights.

| TensorFlow Version | API |
| --- | --- |
| 1.12     | `tf.contrib.lite.TFLiteConverter` |
| 1.9-1.11 | `tf.contrib.lite.TocoConverter`   |
| 1.7-1.8  | `tf.contrib.lite.toco_convert`    |  


---
## Using Command Line

It is adviseable to use the API instead of the command line tool for TF version `1.9`. TF version `1.8` has no command line tool. In this case too your Keras model should be in `hdf5` format and contain both the weights and architecture.


### Example
```python
tflite_convert \
  --output_file=/tmp/foo.tflite \
  --keras_model_file=/tmp/keras_model.h5
```

Refer to this [link](https://www.tensorflow.org/lite/convert/cmdline_reference) for the complete command line reference.

## Out of the Box Models
Here's a list of [out of the box models](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/models.md) which can be readily deployed in mobile devices.

## Embedding The Model

TensorFlow provides a [step-by-step walkthrough and an Android app skeleton](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite/#3) to just plug and play (or even deploy) with the model, without having any first hand experience of Android Development.
