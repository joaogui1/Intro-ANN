import sklearn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def do_pca(data, projdims):

    #sees if thera are less points than dimensions
    usetransf = data.shape[0] < data.shape[1]

    mean = np.mean(data, 0)
    data = data - mean
    if usetransf:
        cov = np.matmul(data, data.transpose())/(data.shape[0] - 1)
    else:
        cov = np.matmul(data.transpose(), data)/(data.shape[0] - 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    #
    if usetransf:
        proj = np.matmul(data.transpose(), eigvecs[:, 0:projdims])
        # (normalize the projection direction to unit length)
        proj = proj/(1e-6 + np.sqrt(np.sum(np.power(proj, 2), 0)))
    else:
        proj = eigvecs[:, 0:projdims]

    return mean, proj, eigvals

# load the iris dataset
iris = datasets.load_iris()
data = iris['data'].astype(np.float64) # a 150x4 matrix with features
scaler = StandardScaler()
scaler.fit(data)
# data = scaler.transform(data)
labels = iris['target'] # an array with 150 class labels
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

mean, proj, eigvals = do_pca(data, 2)

eigvals /= sum(eigvals)
print(eigvals)
# plt.bar(np.arange(4), eigvals, 1)
# plt.xticks(np.arange(4), (1, 2, 3, 4))
# plt.xlabel('Principal Components')
# plt.ylabel('Variance Explained by Component')
# plt.show()

# for i in range(4):
#     for j in range(i + 1, 4):
#         plt.scatter(data[:, i], data[:, j], c=labels)
#         plt.xlabel(features[i])
#         plt.ylabel(features[j])
#         plt.show()

# project the features to two dimensions
data = np.matmul(data-mean, proj)
# plot the results 8we use class labels for colorization)

plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
