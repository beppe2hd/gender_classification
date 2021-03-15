import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # DIsable all the TensorFlow Extra Debug Notification.

# set of imports for tensorflow and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
#from tensorflow.keras.models import Sequential

import pathlib # lib for analysis on the dataset folder
from pathlib import Path

import numpy as np

# Training data
data_dir = "./archive/Training";
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*.jpg')))
print(image_count)

# Defining parameters for the image processing
batch_size = 32
# image dimension are fixed. The employed dataset use cropped image sized (114*92)
img_height = 114
img_width = 92

train_set = tf.keras.preprocessing.image_dataset_from_directory(
  "archive/Training",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

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

class_names = train_set.class_names
print(class_names)
num_classes = len(class_names)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_set = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_set = val_set.cache().prefetch(buffer_size=AUTOTUNE)


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
model.add(layers.Dense(num_classes, name="denseClasses"))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=5

history = model.fit(
  train_set,
  validation_data=val_set,
  epochs=epochs
)

model.summary()
model.save("trained_model")

print("The training process is finished")