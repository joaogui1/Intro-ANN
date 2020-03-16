import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import scipy
dataset = []
class_names = ['airplanes', 'cars', 'birds', 'cats', 'deers', 'dogs', 'frogs', 'horses', 'ships', 'trucks']

for classe in class_names:
    dataset.append(scipy.ndimage.imread("dataset/" + classe + ".jpg"))

# dataset.append(scipy.ndimage.imread("plane.jpg"))
# dataset.append(scipy.ndimage.imread("car.jpg"))
# dataset.append(scipy.ndimage.imread("bird.jpg"))
# dataset.append(scipy.ndimage.imread("cat.jpg"))
# dataset.append(scipy.ndimage.imread("deer.jpg"))
# dataset.append(scipy.ndimage.imread("dog.jpg"))
# dataset.append(scipy.ndimage.imread("frog.jpg"))
# dataset.append(scipy.ndimage.imread("horse.jpg"))
# dataset.append(scipy.ndimage.imread("ship.jpg"))
# dataset.append(scipy.ndimage.imread("truck.jpg"))

dataset = np.asarray(dataset).astype('float32')
dataset /= 255.0
labels = range(0, 10)
#labels = keras.utils.to_categorical(labels, 10)


model = keras.models.load_model('cifar.h5')
predictions = model.predict(dataset)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 2
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, labels, dataset)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, labels)
plt.show()
