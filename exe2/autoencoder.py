import csv
import numpy as np

class autoencoder(object):

    def __init__(self, nvis = 10, nhidden = 3, W1 = None, W2 = None, b1 = None, b2 = None, eta = 0.1):

        if W1 == None:
            self.W1 = np.random.rand(nvis, nhidden)
        if W2 == None:
            self.W2 = np.random.rand(nhidden, nvis)
        if b1 == None:
            self.b1 = np.zeros(nhidden)
        if b2 == None:
            self.b2 = np.zeros(nvis)

        self.eta = eta
        self.nvis = nvis
        self.nhidden = nhidden

    def fnet(self, x):
        return 1.0/(1 + np.exp(-x))
    def act_derivative(self, x):
        return x*(1.0 - x)

    def encode(self, x):
        return self.fnet(np.dot(self.W1.T, x) + self.b1)

    def decode(self, x):
        return self.fnet(np.dot(self.W2.T, x) + self.b2)

    def calc_loss(self, x, x_hat):
        diff = x - x_hat
        loss = 0.0
        for i in diff:
            loss += i*i
        return (loss)*0.5

    def grad_descent(self, batch, size):
        loss = 0
        gradW1 = np.zeros(self.W1.shape)
        gradW2 = np.zeros(self.W2.shape)
        gradb1 = np.zeros(self.b1.shape)
        gradb2 = np.zeros(self.b2.shape)

        for x in batch:
            y = self.encode(x)
            x_hat = self.decode(y)

            loss += self.calc_loss(x, x_hat)

            delta2 = x_hat - x

            gradW2 += np.outer(delta2, y).T
            gradb2 += delta2

            delta1 = np.dot(self.W2, delta2)*self.fnet(y)
            gradW1 += np.outer(delta1, x).T
            gradb1 += delta1

        loss /= len(batch)
        gradW1 /= len(batch)
        gradW2 /= len(batch)
        gradb1 /= len(batch)
        gradb2 /= len(batch)

        return loss, gradW1, gradW2, gradb1, gradb2


    def train(self, X, epochs = 100, batch_sz = 1):
        nbatch = int(len(X)/batch_sz)
        for epoch in range(epochs):
            total_loss = 0.0

            for i in range(nbatch):
                batch = X[i*batch_sz : (i + 1)*batch_sz]

                loss, gradW1, gradW2, gradb1, gradb2 = self.grad_descent(batch, len(X))

                total_loss += loss
                self.W1 -= self.eta*gradW1
                self.W2 -= self.eta*gradW2
                self.b1 -= self.eta*gradb1
                self.b2 -= self.eta*gradb2


    def test(self, X):
        avg_loss = 0.0

        for element in X:
            loss = self.calc_loss(element, self.decode(self.encode(element)))
            avg_loss += loss
        print(avg_loss)/len(X)
