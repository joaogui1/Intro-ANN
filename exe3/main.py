import os
import json as js
import numpy as np
import matplotlib.pyplot as mpl
from sklearn import preprocessing


x_ax = range(1, 31)
with open("anndata", 'r') as inp:
    accuracy = js.load(inp)
    mpl.plot(x_ax, accuracy[0], label='MLP Train')
    mpl.plot(x_ax, accuracy[1], label='MLP Test')
    mpl.title("MLP vs RBF performance")
    mpl.ylabel("Accuracy")
    mpl.xlabel("Number of epochs")
    mpl.ylim(0.0, 1.0)

with open("rbfdata", 'r') as inp:
    accuracy = js.load(inp)
    mpl.plot(x_ax, accuracy[0], label='RBF Train')
    mpl.plot(x_ax, accuracy[1], label='RBF Test')

mpl.legend()
mpl.show()
