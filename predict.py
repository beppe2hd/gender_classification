'''
The code analyze a face image and predict its class [male, female]
INPUT argument: path of the image under test
OUTPUT: the predicted class and associate prediction confidence
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable all the TensorFlow Extra Debug Notification.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys

# Defining parameters for the image processing
img_height = 114
img_width = 92

class_names = ['female', 'male']  # classes name definition (coherently with the class retrieved in training)

# Model loading
try:
    model = keras.models.load_model('trained_model_net_1')
except OSError:
    print("The model is not availabe.")
    sys.exit(1)


# Retrieving the image path from command line
if len(sys.argv)==2:
    file_path = sys.argv[1]
    #print("I'll try to load and process your file: ", file_path)
else:
    print("Wrong number of arguments.")
    sys.exit(1)

# the file is loaded and scaled/reshaped if necessary
try:
    image = keras.preprocessing.image.load_img(
      file_path, target_size=(img_height, img_width)
    )
except FileNotFoundError:
    print("The file is not available.")
    sys.exit(1)

 # Image is prepared for the analysis
img_for_processing = keras.preprocessing.image.img_to_array(image)
img_for_processing = tf.expand_dims(img_for_processing, 0)

# The prediction and its percent confidence (score) are computed
predictions = model.predict(img_for_processing)
score = tf.nn.softmax(predictions[0]) # the score vector refers to the class order

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)