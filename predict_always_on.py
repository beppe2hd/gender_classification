import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # DIsable all the TensorFlow Extra Debug Notification.

import tensorflow as tf
#from tensorflow.keras import datasets, layers, models
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
import PIL
import sys #for the argument from command line

def get_classification(img_path):
    # the file is loaded and scaled/reshaped if necessary
    try:
        image = keras.preprocessing.image.load_img(
            img_path, target_size=(img_height, img_width)
        )
    except FileNotFoundError:
        print("The file is not availabe.")
        return

    img_for_processing = keras.preprocessing.image.img_to_array(image)  # the file is loaded and reshaped if necessary
    img_for_processing = tf.expand_dims(img_for_processing, 0)  # Image is prepared for the analysis

    # The prediction and its score are computed
    predictions = model.predict(img_for_processing)
    score = tf.nn.softmax(predictions[0])  # the score vector refers to the class order

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    return


# Defining parameters for the image processing
img_height = 114
img_width = 92

class_names = ['female', 'male'] # classes name definition (coherently with the class retrieved in training)

# Model Loading
try:
    model = keras.models.load_model('trained_model_net_1') # the model is loaded
except OSError:
    print("The model is not availabe.")
    sys.exit(1)


# Retrieving the image path from command line
while True:
    input_file = input("Please, provide the image path or 'q' to exit: ")

    if input_file == 'q':
        break

    get_classification(input_file)

