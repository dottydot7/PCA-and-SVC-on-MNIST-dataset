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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

path = ""

#MNIST
test_images = idx2numpy.convert_from_file(path + "t10k-images-idx3-ubyte")
test_images_label = idx2numpy.convert_from_file(path + "t10k-labels-idx1-ubyte").reshape(-1)
train_images = idx2numpy.convert_from_file(path + "train-images-idx3-ubyte")
train_images_label = idx2numpy.convert_from_file(path + "train-labels-idx1-ubyte").reshape(-1)

# Flatten each image
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

# Standardize
scalar = StandardScaler()
train_images_standardized = scalar.fit_transform(train_images)
test_images_standardized = scalar.transform(test_images)

#Fashion MNIST
fashion_test_images = idx2numpy.convert_from_file(path + "fashion_t10k-images-idx3-ubyte")
fashion_test_images_label = idx2numpy.convert_from_file(path +"fashion_t10k-labels-idx1-ubyte").reshape(-1)
fashion_train_images = idx2numpy.convert_from_file(path + "fashion_train-images-idx3-ubyte")
fashion_train_images_label = idx2numpy.convert_from_file(path + "fahion_train-labels-idx1-ubyte").reshape(-1)

# Flatten each image
fashion_train_images = fashion_train_images.reshape(fashion_train_images.shape[0], -1)
fashion_test_images = fashion_test_images.reshape(fashion_test_images.shape[0], -1)

# Standardize
fashion_train_images_standardized = scalar.fit_transform(fashion_train_images)
fashion_test_images_standardized = scalar.transform(fashion_test_images)

# --- Dimensionality Reduction Functions ---
def apply_pca(X_train, X_test, n_components):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

def apply_lda(X_train, X_test, y_train, n_components):
    lda = LDA(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    return X_train_lda, X_test_lda

# MNIST PCA
mnist_pca_50_train, mnist_pca_50_test = apply_pca(train_images_standardized, test_images_standardized, 50)
mnist_pca_100_train, mnist_pca_100_test = apply_pca(train_images_standardized, test_images_standardized, 100)
mnist_pca_200_train, mnist_pca_200_test = apply_pca(train_images_standardized, test_images_standardized, 200)

# MNIST LDA components not working for 50, 100, 200
mnist_lda_train, mnist_lda_test = apply_lda(train_images_standardized, test_images_standardized, train_images_label, 9)

#Fashion PCA
fashion_pca_50_train, fashion_pca_50_test = apply_pca(fashion_train_images_standardized, fashion_test_images_standardized, 50)
fashion_pca_100_train, fashion_pca_100_test = apply_pca(fashion_train_images_standardized, fashion_test_images_standardized, 100)
fashion_pca_200_train, fashion_pca_200_test = apply_pca(fashion_train_images_standardized, fashion_test_images_standardized, 200)

# Fashion LDA same issue as prev mnist lda
fashion_lda_train, fashion_lda_test = apply_lda(fashion_train_images_standardized, fashion_test_images_standardized, fashion_train_images_label, 9)
