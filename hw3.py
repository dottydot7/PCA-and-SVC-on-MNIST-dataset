import sys
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
#from distutils.version import LooseVersion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC as sklsvc
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.metrics import mean_squared_error
import idx2numpy 
from sklearn.decomposition import PCA

path = ""

# MNIST
test_images = idx2numpy.convert_from_file(path + "t10k-images-idx3-ubyte")
test_images_label = idx2numpy.convert_from_file(path + "t10k-labels-idx1-ubyte")
train_images = idx2numpy.convert_from_file(path + "train-images-idx3-ubyte")
train_images_label = idx2numpy.convert_from_file(path + "train-labels-idx1-ubyte")

# Flatten each image
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

# Standardize
scalar = StandardScaler()
train_images_standardized = scalar.fit_transform(train_images)
test_images_standardized = scalar.transform(test_images)

# Fashion MNIST
fashion_test_images = idx2numpy.convert_from_file(path + "fashion_t10k-images-idx3-ubyte")
fashion_test_images_label = idx2numpy.convert_from_file(path +"fashion_t10k-labels-idx1-ubyte")
fashion_train_images = idx2numpy.convert_from_file(path + "fashion_train-images-idx3-ubyte")
fashion_train_images_label = idx2numpy.convert_from_file(path + "fahion_train-labels-idx1-ubyte")

# Flatten each image
fashion_train_images = fashion_train_images.reshape(fashion_train_images.shape[0], -1)
fashion_test_images = fashion_test_images.reshape(fashion_test_images.shape[0], -1)

# Standardize
fashion_train_images_standardized = scalar.fit_transform(fashion_train_images)
fashion_test_images_standardized = scalar.transform(fashion_test_images)

# --- PCA Utility Functions ---
def plot_pca_variance(pca, dataset_name, n_components):
    eigen_vals = pca.explained_variance_
    tot = sum(eigen_vals)
    
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, n_components + 1), var_exp, align='center',
            label='Individual explained variance')
    plt.step(range(1, n_components + 1), cum_var_exp, where='mid',
             label='Cumulative explained variance', color='red')
    
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.title(f'{dataset_name} PCA Variance (n_components={n_components})')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def apply_pca(X_train, X_test, n_components, dataset_name="Dataset"):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    var_ratio = sum(pca.explained_variance_ratio_)
    print(f"{dataset_name} PCA (n_components={n_components}) Variance Explained: {var_ratio:.4f}")
    plot_pca_variance(pca, dataset_name, n_components)
    return X_train_pca, X_test_pca, var_ratio

# --- Run PCA for Required Components ---
for n in [50, 100, 200]:
    mnist_pca_train, mnist_pca_test, _ = apply_pca(train_images_standardized, test_images_standardized, n, dataset_name="MNIST")
    fashion_pca_train, fashion_pca_test, _ = apply_pca(fashion_train_images_standardized, fashion_test_images_standardized, n, dataset_name="Fashion MNIST")
