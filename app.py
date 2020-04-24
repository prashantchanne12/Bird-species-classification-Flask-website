from __future__ import division, print_function
import sys
import os
import glob
import re
import tensorflow as tf
import tensorflow_hub as hub
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
# Define a flask app
app = Flask(__name__)
unique_labels=['ALBATROSS', 'ALEXANDRINE PARAKEET', 'AMERICAN AVOCET',
       'AMERICAN BITTERN', 'AMERICAN COOT', 'AMERICAN GOLDFINCH',
       'AMERICAN KESTREL', 'AMERICAN PIPIT', 'AMERICAN REDSTART',
       'ANHINGA', 'ANNAS HUMMINGBIRD', 'ANTBIRD', 'ARARIPE MANAKIN',
       'BALD EAGLE', 'BALTIMORE ORIOLE', 'BANANAQUIT',
       'BAR-TAILED GODWIT', 'BARN OWL', 'BARN SWALLOW',
       'BAY-BREASTED WARBLER', 'BELTED KINGFISHER', 'BIRD OF PARADISE',
       'BLACK FRANCOLIN', 'BLACK SKIMMER', 'BLACK SWAN',
       'BLACK THROATED WARBLER', 'BLACK VULTURE',
       'BLACK-CAPPED CHICKADEE', 'BLACK-NECKED GREBE',
       'BLACKBURNIAM WARBLER', 'BLUE GROUSE', 'BLUE HERON', 'BOBOLINK',
       'BROWN THRASHER', 'CACTUS WREN', 'CALIFORNIA CONDOR',
       'CALIFORNIA GULL', 'CALIFORNIA QUAIL', 'CANARY',
       'CAPE MAY WARBLER', 'CARMINE BEE-EATER', 'CASPIAN TERN',
       'CASSOWARY', 'CHARA DE COLLAR', 'CHIPPING SPARROW',
       'CINNAMON TEAL', 'COCK OF THE  ROCK', 'COCKATOO', 'COMMON GRACKLE',
       'COMMON HOUSE MARTIN', 'COMMON LOON', 'COMMON POORWILL',
       'COMMON STARLING', 'COUCHS KINGBIRD', 'CRESTED AUKLET',
       'CRESTED CARACARA', 'CROW', 'CROWNED PIGEON', 'CUBAN TODY',
       'CURL CRESTED ARACURI', 'D-ARNAUDS BARBET', 'DARK EYED JUNCO',
       'DOVEKIE', 'DOWNY WOODPECKER', 'EASTERN BLUEBIRD',
       'EASTERN MEADOWLARK', 'EASTERN ROSELLA', 'EASTERN TOWEE',
       'ELEGANT TROGON', 'ELLIOTS  PHEASANT', 'EMPEROR PENGUIN', 'EMU',
       'EURASIAN MAGPIE', 'EVENING GROSBEAK', 'FLAME TANAGER', 'FLAMINGO',
       'FRIGATE', 'GILA WOODPECKER', 'GLOSSY IBIS', 'GOLD WING WARBLER',
       'GOLDEN CHLOROPHONIA', 'GOLDEN EAGLE', 'GOLDEN PHEASANT',
       'GOULDIAN FINCH', 'GRAY CATBIRD', 'GRAY PARTRIDGE', 'GREEN JAY',
       'GREY PLOVER', 'GUINEAFOWL', 'HAWAIIAN GOOSE', 'HOODED MERGANSER',
       'HOOPOES', 'HORNBILL', 'HOUSE FINCH', 'HOUSE SPARROW',
       'HYACINTH MACAW', 'INCA TERN', 'INDIGO BUNTING', 'JABIRU',
       'JAVAN MAGPIE', 'KILLDEAR', 'KING VULTURE', 'LARK BUNTING',
       'LILAC ROLLER', 'LONG-EARED OWL', 'MALEO', 'MALLARD DUCK',
       'MANDRIN DUCK', 'MARABOU STORK', 'MASKED BOOBY',
       'MIKADO  PHEASANT', 'MOURNING DOVE', 'MYNA', 'NICOBAR PIGEON',
       'NORTHERN CARDINAL', 'NORTHERN FLICKER', 'NORTHERN GANNET',
       'NORTHERN GOSHAWK', 'NORTHERN JACANA', 'NORTHERN MOCKINGBIRD',
       'NORTHERN RED BISHOP', 'OCELLATED TURKEY', 'OSPREY', 'OSTRICH',
       'PAINTED BUNTIG', 'PARADISE TANAGER', 'PARUS MAJOR', 'PEACOCK',
       'PELICAN', 'PEREGRINE FALCON', 'PINK ROBIN', 'PUFFIN',
       'PURPLE FINCH', 'PURPLE GALLINULE', 'PURPLE MARTIN',
       'PURPLE SWAMPHEN', 'QUETZAL', 'RAINBOW LORIKEET',
       'RED FACED CORMORANT', 'RED HEADED WOODPECKER',
       'RED THROATED BEE EATER', 'RED WINGED BLACKBIRD',
       'RED WISKERED BULBUL', 'RING-NECKED PHEASANT', 'ROADRUNNER',
       'ROBIN', 'ROCK DOVE', 'ROSY FACED LOVEBIRD', 'ROUGH LEG BUZZARD',
       'RUBY THROATED HUMMINGBIRD', 'RUFOUS KINGFISHER', 'RUFUOS MOTMOT',
       'SAND MARTIN', 'SCARLET IBIS', 'SCARLET MACAW', 'SHOEBILL',
       'SNOWY EGRET', 'SORA', 'SPLENDID WREN', 'SPOONBILL',
       'STORK BILLED KINGFISHER', 'STRAWBERRY FINCH', 'TAIWAN MAGPIE',
       'TEAL DUCK', 'TIT MOUSE', 'TOUCHAN', 'TRUMPTER SWAN',
       'TURKEY VULTURE', 'TURQUOISE MOTMOT', 'VARIED THRUSH',
       'VENEZUELIAN TROUPIAL', 'VERMILION FLYCATHER',
       'VIOLET GREEN SWALLOW', 'WESTERN MEADOWLARK',
       'WHITE CHEEKED TURACO', 'WHITE TAILED TROPIC', 'WILD TURKEY',
       'WILSONS BIRD OF PARADISE', 'WOOD DUCK', 'YELLOW HEADED BLACKBIRD',
       'INDIAN CROW', 'INDIAN SPARROW', 'PIGEON']

def load_model(model_path):
  print(f'Loading model from :{model_path}')
  model=tf.keras.models.load_model(model_path,
                                   custom_objects={
                                       'KerasLayer':hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v2/classification/4")

                                   })
  return model

def get_pred_label(prediction_probability,unique_labels):
  return unique_labels[np.argmax(prediction_probability)]  

def get_pred_get_pred(custom_preds):
    custom_pred_label=[get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
    custom_pred_label

def get_image_label(image_path,label):
  image=preprocess_image(image_path)
  return image,label

# define image size
IMG_SIZE=224
BATCH_SIZE=32

def preprocess_image(image_path,img_size=224):
  image=tf.io.read_file(image_path)
  image=tf.image.decode_jpeg(image,channels=3)
  image=tf.image.convert_image_dtype(image,tf.float32)
  image = tf.image.resize(image,size=[IMG_SIZE,IMG_SIZE])
  return image


def create_data_batches(x,batch_size=32):
    print('Creating test data branches....')
    x=[x]
    data=tf.data.Dataset.from_tensor_slices((tf.constant(x)))
    data_batch=data.map(preprocess_image).batch(BATCH_SIZE)
    return data_batch


def model_predict(custom_images_path,loaded_full_model):
    custom_data=create_data_batches(custom_images_path)
    custom_preds = loaded_full_model.predict(custom_data)
    custom_pred_label=[get_pred_label(custom_preds[i],unique_labels) for i in range(len(custom_preds))]
    return custom_pred_label


loaded_full_model=load_model('model/fullmodel.h5')
@app.route('/')
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path,loaded_full_model)            
        return result[0]
    return None


if __name__ == '__main__':
    app.run(debug=True)
