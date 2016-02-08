import keras.caffe.convert as convert
import argparse

""" 

    USAGE EXAMPLE

	    python caffe2keras.py 	-load_path '/media/HDD_2TB/CNN_MODELS/GoogleNet-Food101/' -prototxt 'train_val_finetunning_for_keras.prototxt' -caffemodel 'foodRecognition_googlenet_finetunning_v2_1_iter_448000.caffemodel'

"""

parser = argparse.ArgumentParser(description='Converts a Caffe model to Keras.')
parser.add_argument('-load_path', type=str,
                   help='path where both the .prototxt and the .caffemodel files are stored')
parser.add_argument('-prototxt', type=str,
                   help='name of the .prototxt file')
parser.add_argument('-caffemodel', type=str,
                   help='name of the .caffemodel file')
parser.add_argument('-store_path', type=str, default='',
                   help='path to the folder where the Keras model will be stored (default: -load_path).')
parser.add_argument('-debug', action='store_true', default=0,
		   help='use debug mode')

args = parser.parse_args()


def main(args):
    if(not args.store_path):
    	store_path = args.load_path

    print "Converting model..."
    model = convert.caffe_to_keras(args.load_path+'/'+args.prototxt, args.load_path+'/'+args.caffemodel, debug=args.debug)
    print "Finished converting model."
    
    # Save converted model structure
    print "Storing model..."
    json_string = model.to_json()
    open(store_path + '/Keras_model_structure.json', 'w').write(json_string)
    # Save converted model weights
    model.save_weights(store_path + '/Keras_model_weights.h5', overwrite=True)
    print "Finished storing the converted model to "+ store_path

main(args)

