# PROGRAMMER: Jeff Carter
# DATE CREATED: 2020-12-30
# REVISED DATE: 2020-12-31
# PURPOSE: Uses a pre-trained neural network to identify the type of flower captured in an image

import json

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from argparse import ArgumentParser
from os import linesep, path
from PIL import Image

def main():
    '''
    The main method of predict.py
    '''
    
    args = get_args()
    
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    
    processed_image = process_image(args.image_path)
    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    
    probs, classes = predict(processed_image, model, args.top_k)
    for idx in range(len(classes)):
        name = class_names[classes[idx]]
        print(linesep)
        print('Prediction #{}:'.format(idx+1))
        print('  Name: {}'.format(name))
        print('  Probability: {:.5f}'.format(probs[idx]))


def get_args():
    '''
    Gets the arguments that will be used by the script
    '''
    
    DEFAULT_TOP_K = 1
    DEFAULT_CAT_NAMES = './label_map.json'
    
    parser = ArgumentParser(description='Accepts options related to prediction output')
    
    parser.add_argument('image_path')
    parser.add_argument('model_path')
    parser.add_argument('--top_k', type=int, default=DEFAULT_TOP_K,
                        help='Specifies the number of values to return, e.g., the top 3 predictions will be returned if the value is 3. Default value is {}.'.format(DEFAULT_TOP_K))
    parser.add_argument('--category_names', default=DEFAULT_CAT_NAMES, help='Can be used to specify your own category names file.')
    
    args = parser.parse_args()
    validate_args(args)
    
    return args


def validate_args(args):
    '''
    Checks that the args are valid
    '''
    
    NUM_CATEGORIES = 102
    
    if not path.isfile(args.image_path):
        raise ValueError('Specified image, {}, not found'.format(args.image_path))
        
    if not path.isfile(args.model_path):
        raise ValueError('Specified model, {}, not found'.format(args.model_path))
    
    if not path.isfile(args.category_names):
        raise ValueError('category names file, {}, not found'.format(args.category_names))
    
    if args.top_k < 1 or args.top_k > NUM_CATEGORIES:
        raise ValueError('The value specified for top_k must be a positive number that is less than the the number of categories ({})'.format(NUM_CATEGORIES))


def process_image(image_path):
    '''
    processes the image so that it can work properly with the neural network
    '''
    
    IMG_SIZE = 224
    
    im = Image.open(image_path)
    image = np.asarray(im)
    
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image /=255
    
    return image.numpy()


def predict(processed_image, model, top_k):
    '''
    predicts which classes the image most likely belongs to and
    returns the class numbers and their respective probability values
    '''
    
    prediction_batch = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(prediction_batch)[0]
    
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    top_probs = list()
    for idx in range(top_k):
        top_probs.append( predictions[top_indices[idx]] )
        
    top_classes = top_indices + 1
    
    return top_probs, top_classes.astype(np.str).tolist()


if __name__ == "__main__":
    main()
