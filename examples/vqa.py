'''Visual Question Answer with Keras

Before running this script, download the following models:

* VGG16 model at https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
* VQA_weights and label_encoder at https://github.com/mynameisvinn/VQA_Demo/tree/master/models

Make sure the variables 'VQA_weights_file_name', 'label_encoder_file_name', and 'CNN_weights_file_name' in this script matches the location of the files.

Then, run the script with:

```
python demo.py -image_file_name path_to_file -question "Question to be asked"
```

e.g
```
python demo.py -image_file_name image.jpg -question "Where is the cat?"
```


Results are comparable to those provided in Agrawal et al.:
"VQA: Visual Question Answering."
https://arxiv.org/pdf/1505.00468v6.pdf

'''

import os, argparse
import cv2, spacy, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers.core import Reshape, Activation, Dropout, Dense, Flatten
from keras.layers import LSTM, Merge, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D




VQA_weights_file_name   = 'models/VQA/VQA_MODEL_WEIGHTS.hdf5' # update to file location 
label_encoder_file_name = 'models/VQA/FULL_labelencoder_trainval.pkl' # update to file location 
CNN_weights_file_name   = 'models/CNN/vgg16_weights.h5' # update to file location 
verbose = 1





def pop_layer(model):
    '''Removes a layer instance on top of the layer stack.

    Popping out layers will be necessary to extract 4096 dimension feature vector from VGG16.
    
    For more information, see https://github.com/fchollet/keras/issues/2371#issuecomment-211734276
    '''
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False

    return model





def VGG_16(weights_path=None):
    '''Define VGG16 architecture in Keras. 

    VGG16 will be used to generate feature vectors for images, not for classification!

    For more information, see https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html


    parameters:
    -----------
    @weights_path: string, representing location of VGG16 weights.

    returns:
    --------
    @model: keras VGG16 with the last two layers (softmax and dropout) removed.
    '''
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu')) # represents 4096 dimension feature vector
    model.add(Dropout(0.5)) 
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    # pop out last two layers (softmax and dropout) to get the last fully connected layer with 4096 dimensions
    model = pop_layer(model)
    model = pop_layer(model)
        
    return model






def get_image_model(CNN_weights_file_name):
    ''' Takes the CNN weights file, and returns the VGG model update with the weights.

    an overview on vgg16 architecture can be found at https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

    parameters:
    -----------
    @CNN_weights_file_name: string, representing file location of VGG16 weights.

    returns:
    --------
    @image_model: keras VGG_16 model.
    '''

    image_model = VGG_16(CNN_weights_file_name)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    image_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    
    return image_model



def get_image_features(image_file_name, CNN_weights_file_name):
    ''' For an given image, extract VGG16 feature vector, which is a single 4096 dimension vector.

    parameters:
    -----------
    @image_file_name: string, representing image fname to be queried.
    @CNN_weights_file_name: string, representing model weights.

    returns:
    --------
    @image_features: numpy array, representing 4096 dimension vector.

    '''
    image_features = np.zeros((1, 4096))

    
    # resize and reshape images to fit VGG16 input dimensions
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0) 

    image_features[0,:] = get_image_model(CNN_weights_file_name).predict(im)[0]

    return image_features



def VQA_MODEL():
    ''' Define fused LSTM/CNN model.

    LSTM and CNN are fused, then an additional 1000-node layer and softmax layer is added.

    '''
    image_feature_size = 4096
    word_feature_size = 300
    number_of_LSTM = 3
    number_of_hidden_units_LSTM = 512
    max_length_questions = 30
    number_of_dense_layers = 3
    number_of_hidden_units = 1024
    activation_function = 'tanh'
    dropout_pct = 0.5


    # Image model
    model_image = Sequential()
    model_image.add(Reshape((image_feature_size,), input_shape=(image_feature_size,)))

    # Language Model
    model_language = Sequential()
    model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True, input_shape=(max_length_questions, word_feature_size)))
    model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True))
    model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=False))

    # combined model
    model = Sequential()
    model.add(Merge([model_language, model_image], mode='concat', concat_axis=1))

    for _ in xrange(number_of_dense_layers):
        model.add(Dense(number_of_hidden_units, init='uniform'))
        model.add(Activation(activation_function))
        model.add(Dropout(dropout_pct))

    model.add(Dense(1000))
    model.add(Activation('softmax'))

    return model





def get_VQA_model(VQA_weights_file_name):
    ''' Given the VQA model and its weights, compiles and returns the model '''
    vqa_model = VQA_MODEL()
    vqa_model.load_weights(VQA_weights_file_name)

    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    return vqa_model

def get_question_features(question):
    ''' For a given question, a unicode string, returns the timeseris vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector '''
    word_embeddings = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in xrange(len(tokens)):
            question_tensor[0,j,:] = tokens[j].vector
    return question_tensor


def main():
    ''' accepts command line arguments for image file and the question and 
    builds the image model (VGG) and the VQA model (LSTM and MLP) 
    prints the top 5 response along with the probability of each '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-image_file_name', type=str, default='test.jpg')
    parser.add_argument('-question', type=str, default='What vechile is in the picture?')
    args = parser.parse_args()

    
    
    image_features = get_image_features(args.image_file_name, CNN_weights_file_name)
    question_features = get_question_features(unicode(args.question, 'utf-8'))
    vqa_model = get_VQA_model(VQA_weights_file_name)
    y_output = vqa_model.predict([question_features, image_features])
    y_sort_index = np.argsort(y_output)

    # This task here is represented as a classification into a 1000 top answers
    # this means some of the answers were not part of trainng and thus would 
    # not show up in the result.
    # These 1000 answers are stored in the sklearn Encoder class
    labelencoder = joblib.load(label_encoder_file_name)
    for label in reversed(y_sort_index[0,-5:]):
        print str(round(y_output[0,label]*100,2)).zfill(5), "% ", labelencoder.inverse_transform(label)

if __name__ == "__main__":
    main()
