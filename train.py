import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # DIsable all the TensorFlow Extra Debug Notification.

# set of imports for tensorflow and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pathlib # library for analysis on the dataset folder
from pathlib import Path

import matplotlib.pyplot as plt # for accuracy training plotting

import numpy as np

# Training data
data_dir = "./archive/Training"
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*.jpg')))
print("The image count is" + image_count)

# Defining parameters for the image processing
batch_size = 32

# Image dimension are fixed. The employed dataset use cropped image sized (114*92)
img_height = 114
img_width = 92

# The dataset is prepared for training and organized in tensors
train_set = tf.keras.preprocessing.image_dataset_from_directory(
  "archive/Training",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# The validation dataset is prepared for testing and organized in tensors
val_set = tf.keras.preprocessing.image_dataset_from_directory(
  "archive/Validation",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

for image_batch, labels_batch in train_set:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# The classes names are retrieved by the train_set object
class_names = train_set.class_names
print(class_names)
num_classes = len(class_names)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_set = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_set = val_set.cache().prefetch(buffer_size=AUTOTUNE)


# The CNN structure is created

# Network1
#
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten(name="flatten"))
# model.add(layers.Dense(768, activation='relu', name="dens_two"))
# model.add(layers.Dense(256, activation='relu', name="dens_three"))
# model.add(layers.Dense(num_classes, name="denseClasses"))

# Network2

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten(name="flatten"))
model.add(layers.Dense(1014, activation='relu', name="dens_one"))
model.add(layers.Dense(768, activation='relu', name="dens_two"))
model.add(layers.Dense(256, activation='relu', name="dens_three"))
model.add(layers.Dense(num_classes, name="denseClasses"))

# the object model is setted up  with optimize, loss function and verification metric
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=10 #Training epoches


# The model is trained and training details stored in history variable
history = model.fit(
  train_set,
  validation_data=val_set,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

print("Accuracy: " + str(acc))
print("Validation accuracy: " + str(val_acc))
print("Loss: " + str(loss))
print("Validation Loss: " + str(val_loss))

# the Trained model is summarized and stored
model.summary()
model.save("trained_model")

print("The training process is finished")