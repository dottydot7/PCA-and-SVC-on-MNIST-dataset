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
path = "\\"

#MNIST
test_images = idx2numpy.convert_from_file(path + "t10k-images-idx3-ubyte")
test_images_label = idx2numpy.convert_from_file(path + "t10k-labels-idx1-ubyte")
train_images = idx2numpy.convert_from_file(path + "train-images-idx3-ubyte")
train_images_label = idx2numpy.convert_from_file(path + "train-labels-idx1-ubyte")
plt.imshow(train_images[0])
plt.show()

#Fashion MNIST
fashion_test_images = idx2numpy.convert_from_file(path + "fashion_t10k-images-idx3-ubyte")
fashion_test_images_label = idx2numpy.convert_from_file(path +"fashion_t10k-labels-idx1-ubyte")
fashion_train_images = idx2numpy.convert_from_file(path + "fashion_train-images-idx3-ubyte")
fashion_train_images_label = idx2numpy.convert_from_file(path + "fahion_train-labels-idx1-ubyte")
plt.imshow(fashion_train_images[0])
plt.show()