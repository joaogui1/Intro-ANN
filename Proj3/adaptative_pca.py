'''PCA network'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

import tensorflow as tf
from tensorflow import keras


def get_data(datafile):
    ''' Returs data as a matrix of features and a target vector'''
    data = pd.read_csv(os.getcwd() + '/' + datafile)
    targets = np.array(data.iloc[:, 0])
    features = np.array(data.iloc[:, 1:])

    onehot_targets = np.zeros((targets.shape[0], 3))
    for i in range(targets.shape[0]):
        onehot_targets[i][targets[i] - 1] = 1
    return features, onehot_targets

def normalize(X):
    '''Normalizes data,  scaling the values to the [0, 1] range'''
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(X)
    return x_scaled

elapsed_epochs = 1
stabilization_cycles = 5
NUMBER_OF_COMPONENTS = 5

train_features, train_targets = get_data('wine.csv')
train_features = normalize(train_features)

test_features, test_targets = get_data('wine_test.csv')
test_features = normalize(test_features)


# initializes weights
w_old = np.ones((train_features.shape[1], NUMBER_OF_COMPONENTS))
w_oja = np.random.normal(scale=0.25, size=(train_features.shape[1], NUMBER_OF_COMPONENTS))
C_oja = np.triu(np.random.uniform(-0.01, 0.01, size=(NUMBER_OF_COMPONENTS, NUMBER_OF_COMPONENTS)))
np.fill_diagonal(C_oja, 0.0)

w_oja /= np.linalg.norm(w_oja)
w_old /= np.linalg.norm(w_old)


while np.linalg.norm(w_oja - w_old) > 0.00001:
    learning_rate = 1.0/(100*elapsed_epochs)
    for x in train_features:
        w_old = w_oja.copy()
        aux_y = np.zeros((NUMBER_OF_COMPONENTS, 1))

        for _ in range(stabilization_cycles):
            predicted_y = np.dot(x, w_oja) + np.dot(C_oja, aux_y)[0]
            aux_y = predicted_y.copy()
        # aux = learning_rate*np.outer(x - np.dot(w_oja, predicted_y), predicted_y)
        w_oja += learning_rate*np.outer(x - np.dot(w_oja, predicted_y), predicted_y)
        C_oja += -learning_rate*np.outer(predicted_y + np.dot(C_oja, predicted_y), predicted_y)

        C_oja = np.tril(C_oja)
        np.fill_diagonal(C_oja, 0.0)

    elapsed_epochs += 1

    # print(np.linalg.norm(w_oja - w_old))
    # print(learning_rate)
    # print(w_oja)
    # A_A = input()

aux_y = np.zeros((train_features.shape[0], NUMBER_OF_COMPONENTS))

for _ in range(stabilization_cycles):
    principal_train = np.asarray([np.dot(x, w_oja) + np.dot(C_oja, y)[0]
                                  for x, y in zip(train_features, aux_y)])
    aux_y = principal_train.copy()

aux_y = np.zeros((test_features.shape[0], NUMBER_OF_COMPONENTS))

for _ in range(stabilization_cycles):
    principal_test = np.asarray([np.dot(x, w_oja) + np.dot(C_oja, y)[0]
                                 for x, y in zip(test_features, aux_y)])
    aux_y = principal_test.copy()


model = keras.Sequential([
    keras.layers.Dense(3, input_dim=13, activation=tf.nn.relu),
    keras.layers.Dense(3, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_features, train_targets,
                    batch_size=32,
                    epochs=60,
                    verbose=1,
                    validation_data=(test_features, test_targets)).history

test_loss, test_acc = model.evaluate(train_features, train_targets)
print(test_loss, test_acc)

plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
