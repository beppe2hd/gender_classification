import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable all the TensorFlow Extra Debug Notification.

# set of imports for tensorflow and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
#from tensorflow.keras.models import Sequential

import pathlib # lib for analysis on the dataset folder
from pathlib import Path

import numpy as np

# function "conf_norm" gat as parameter a np matrix and return a numpy matrix with the same dimension
# normalized (norm L1) row by row
def conf_norm(matrix):
    normalized_matrix = matrix
    for i in range(0, matrix.shape[0]):
        row_sum = np.sum(matrix[i][:])
        for j in range(0, matrix.shape[1]):
            normalized_matrix[i][j] = normalized_matrix[i][j]/row_sum

    return normalized_matrix


# Training data
data_dir = "./archive/Validation"
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*.jpg')))
print(image_count)

model = keras.models.load_model('trained_model')  # The model is loaded
class_names = ['female', 'male']  # classes name definition (coherently with the class retrieved in training)

# Defining parameters for the image processing
# image dimension are fixed. The employed dataset use cropped image sized (114*92)
img_height = 114
img_width = 92

confusion_matrix = np.zeros((2, 2))
print(confusion_matrix)

for path in Path(data_dir).iterdir():
    if path.is_dir():
        category_under_test = str(path).split('/').pop()
        count = 0
        for subpath in Path(path).glob('*.jpg'):
            count +=  1
            if count == 80:
                break
            print(subpath)
            image = keras.preprocessing.image.load_img(
                subpath, target_size = (img_height, img_width)
            )
            img_for_processing = keras.preprocessing.image.img_to_array(image)
            img_for_processing = tf.expand_dims(img_for_processing, 0)
            predictions = model.predict(img_for_processing)
            score = tf.nn.softmax(predictions[0])
            predicted_category = class_names[np.argmax(score)]
            if category_under_test == 'male':
                if category_under_test == predicted_category:
                    confusion_matrix[0][0] += 1
                else:
                    confusion_matrix[0][1] += 1
            if category_under_test == 'female':
                if category_under_test == predicted_category:
                    confusion_matrix[1][1] += 1
                else:
                    confusion_matrix[1][0] += 1


#Confusion matrix reports:
# number of images estimated as male whose GT is male in [0][0]
# number of images estimated as male whose GT is female in [0][1]
# number of images estimated as female whose GT is male in [1][0]
# number of images estimated as female whose GT is female in [1][1]

# Confusion Matrix computed on validation set
print("Confusion Matrix computed on validation set")
print(confusion_matrix)

# Normalized Confusion Matrix computed on validation set
normalized_matrix = conf_norm(confusion_matrix) # normalize the row of the confusion matrix
print("Normalized Confusion Matrix computed on validation set")
print(normalized_matrix)