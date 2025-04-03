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
from distutils.version import LooseVersion
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

# path = "\\"

#MNIST
test_images = idx2numpy.convert_from_file("t10k-images-idx3-ubyte")
test_images_label = idx2numpy.convert_from_file("t10k-labels-idx1-ubyte")
train_images = idx2numpy.convert_from_file("train-images-idx3-ubyte")
train_images_label = idx2numpy.convert_from_file("train-labels-idx1-ubyte")
# plt.imshow(train_images[0]) #expects 2d data, won't work with reshape
# plt.show()

#Fashion MNIST
fashion_test_images = idx2numpy.convert_from_file("fashion_t10k-images-idx3-ubyte")
fashion_test_images_label = idx2numpy.convert_from_file("fashion_t10k-labels-idx1-ubyte")
fashion_train_images = idx2numpy.convert_from_file("fashion_train-images-idx3-ubyte")
fashion_train_images_label = idx2numpy.convert_from_file("fahion_train-labels-idx1-ubyte")
# plt.imshow(fashion_train_images[0])#expects 2d data, won't work with reshape
# plt.show()


def plot_images(images, labels, title, n=10):
    plt.figure(figsize=(15, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

#Plot MNIST and Fashion MNIST raw images
plot_images(train_images, train_images_label, "MNIST Raw Images")
plot_images(fashion_train_images, fashion_train_images_label, "Fashion MNIST Raw Images")

#flatten images
mnist_train_images_flat = train_images.reshape(train_images.shape[0], -1)
mnist_test_images_flat = test_images.reshape(test_images.shape[0], -1)
fashion_train_images_flat = fashion_train_images.reshape(fashion_train_images.shape[0], -1)
fashion_test_images_flat = fashion_test_images.reshape(fashion_test_images.shape[0], -1)

#before and after flattening 
print("Original MNIST Train Images Shape: ", train_images.shape) # (60000, 28, 28) 
print("Flattened MNIST Train Images Shape:", mnist_train_images_flat.shape) # (60000, 784) 
print("Original Fashion MNIST Train Images Shape:", fashion_train_images.shape) # (60000, 28, 28) 
print("Flattened Fashion MNIST Train Images Shape:", fashion_train_images_flat.shape) # (60000, 784) 
