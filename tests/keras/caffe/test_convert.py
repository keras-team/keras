from __future__ import print_function
import pytest
import keras.caffe.convert as convert

def test_convertGoogleNet():
    load_path = 'keras/caffe/models'
    store_path = 'keras/caffe/models'
    prototxt = 'train_val_for_keras.prototxt'
    caffemodel = 'bvlc_googlenet.caffemodel'

    # Convert model from caffe to keras
    model = convert.caffe_to_keras(load_path+'/'+prototxt, load_path+'/'+caffemodel, debug=False)
    assert(model.__class__.__name__ == 'Model')

    # Save converted model structure
    json_string = model.to_json()
    open(store_path + '/Keras_model_structure.json', 'w').write(json_string)
    # Save converted model weights
    model.save_weights(store_path + '/Keras_model_weights.h5', overwrite=True)


if __name__ == '__main__':
    pytest.main([__file__]) 
    # test_convertGoogleNet()
