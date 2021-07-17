##Flower_Classification_WebApp.ipynb


#Original file is located at
    #https://colab.research.google.com/drive/1u2TQs95iQ7FQYU0bEAI2cAgjdi9wrV2T


#Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#Download & Unzip Dataset
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

#Image Count
image_count = len(list(data_dir.glob('*/*')))
print(image_count)

# Roses folder contents  
roses = list(data_dir.glob('roses/*'))
print(roses[0]) # Path of the image
PIL.Image.open(str(roses[0]))# Open a sample image in the roses folder

##Training & Validation Dataset Creation*"""
img_height,img_width=180,180
batch_size=32

# Training Dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#Validation Dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

##Classes & Sample Visualization

#Classes Identification
class_names = train_ds.class_names
class_names

# Visualization
import matplotlib.pyplot as plt
image_batch, label_batch = next(iter(train_ds))# Load data by iter() & going to the next image by next()
plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label = label_batch[i]
  plt.title(class_names[label])
  plt.axis("off")

#Model Making

num_classes = 5

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes,activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

#Save Model
tf.keras.models.save_model(model,'my_model2.hdf5')