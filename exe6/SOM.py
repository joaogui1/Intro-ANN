import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

import tensorflow as tf

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


def _neuron_locations(m, n):
    """
    Yields one by one the 2-D locations of the individual neurons
    in the SOM.
    """
    #Nested iterations over both dimensions
    #to generate all 2-D locations in the map
    for i in range(m):
        for j in range(n):
            yield np.array([i, j])


class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """

    #To check if the SOM has been trained
    is_trained = False

    def __init__(self, m, n, dim, epochs=100, alpha=None, sigma=None):
        """
        Initializes all necessary components of the TensorFlow
        Graph.

        m X n are the dimensions of the SOM. 'epochs' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        """

        #Assign required variables first
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._epochs = abs(int(epochs))

        self._graph = tf.Graph()

        with self._graph.as_default():

            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE

            #Randomly initialized weights vectors for all neurons,
            self.weights = tf.Variable(tf.random_normal(
                [m*n, dim]))

            #Matrix of size [m*n, 2] for SOM grid locations
            #of neurons
            self._location_vects = tf.constant(np.array(
                list(_neuron_locations(m, n))))

            ##PLACEHOLDERS FOR TRAINING INPUTS
            #We need to assign them as attributes to self, since they
            #will be fed in during training

            #The training vector
            self._vect_input = tf.placeholder("float", [dim])
            #Iteration number
            self._iter_input = tf.placeholder("float")

            #closest map vector
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self.weights, tf.stack(
                    [self._vect_input for i in range(m*n)])), 2), 1)),
                                  0)

            #This will extract the location of the BMU based on the BMU's
            #index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]))),
                                 [2])

            #To compute the alpha and sigma values based on iteration
            #number
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input,
                                                       self._epochs))
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)

            #Construct the op that will generate a vector with learning
            #rates for all neurons, based on iteration number and location
            #wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vects, tf.stack(
                    [bmu_loc for i in range(m*n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)

            #Finally, the op that will use learning_rate_op to update
            #the weightage vectors of all neurons based on a particular
            #input
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                                                 for i in range(m*n)])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vect_input for i in range(m*n)]),
                            self.weights))
            new_weightages_op = tf.add(self.weights,
                                       weightage_delta)
            self._training_op = tf.assign(self.weights,
                                          new_weightages_op)

            ##INITIALIZE SESSION
            self._sess = tf.Session()

            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """

        #Training iterations
        for iter_no in range(self._epochs):
            #Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})

        #Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self.weights))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid

        self.is_trained = True

    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self.is_trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """

        if not self.is_trained:
            raise ValueError("SOM not trained yet")

        organized_map = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect-
                                                         self._weightages[x]))
            organized_map.append(self._locations[min_index])

        return organized_map

train_features, train_targets = get_data('wine.csv')
train_features = normalize(train_features)

NUM_ROWS = 13
NUM_COLUMNS = 13
MAX_DIFF = 0.0

self_organizing = SOM(NUM_ROWS, NUM_COLUMNS, 13, epochs =1000)
self_organizing.train(train_features)
centroids = np.asarray(self_organizing.get_centroids())
SO_map = self_organizing.map_vects(train_features)

image_grid = np.zeros((NUM_ROWS, NUM_COLUMNS, 3))

'''The commented code below was used to generate the association map'''
for idx, inp in enumerate(train_features):
    idx_min = np.argmin([np.linalg.norm(node - inp) for line in centroids for node in line])
    idx_min = np.unravel_index(idx_min, (NUM_COLUMNS, NUM_ROWS))
    image_grid[idx_min] += train_targets[idx]

for idl, _ in enumerate(image_grid):
    for idc, _ in enumerate(image_grid[idl]):
        if np.linalg.norm(image_grid[idl][idc]) > 0:
            image_grid[idl][idc] /= np.linalg.norm(image_grid[idl][idc])

for idl, _ in enumerate(image_grid):
    for idc, _ in enumerate(image_grid[idl]):
        if np.linalg.norm(image_grid[idl][idc]) < 1e-5:
            cont = 0
            if idl > 0:
                cont += 1
                image_grid[idl][idc] += image_grid[idl - 1][idc]
            if idc > 0:
                cont += 1
                image_grid[idl][idc] += image_grid[idl][idc - 1]
            if idl + 1 < NUM_ROWS:
                cont += 1
                image_grid[idl][idc] += image_grid[idl + 1][idc]
            if idc + 1 < NUM_COLUMNS:
                cont += 1
                image_grid[idl][idc] += image_grid[idl][idc + 1]
            image_grid[idl][idc] /= cont


plt.imshow(image_grid)
plt.show()
