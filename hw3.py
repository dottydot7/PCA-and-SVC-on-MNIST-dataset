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
from sklearn.decomposition import PCA
import idx2numpy 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# path = "\\"

#MNIST
mnist_test_images = idx2numpy.convert_from_file("t10k-images-idx3-ubyte")
mnist_test_images_label = idx2numpy.convert_from_file("t10k-labels-idx1-ubyte")
mnist_train_images = idx2numpy.convert_from_file("train-images-idx3-ubyte")
mnist_train_images_label = idx2numpy.convert_from_file("train-labels-idx1-ubyte")

#Fashion MNIST
fashion_test_images = idx2numpy.convert_from_file("fashion_t10k-images-idx3-ubyte")
fashion_test_images_label = idx2numpy.convert_from_file("fashion_t10k-labels-idx1-ubyte")
fashion_train_images = idx2numpy.convert_from_file("fashion_train-images-idx3-ubyte")
fashion_train_images_label = idx2numpy.convert_from_file("fahion_train-labels-idx1-ubyte")


# def plot_images(images, labels, title, n=10):
#     plt.figure(figsize=(15, 3))
#     for i in range(n):
#         plt.subplot(1, n, i+1)
#         plt.imshow(images[i], cmap='gray')
#         plt.title(f"Label: {labels[i]}")
#         plt.axis('off')
#     plt.suptitle(title)
#     plt.show()

# #Plot MNIST and Fashion MNIST raw images
# plot_images(train_images, train_images_label, "MNIST Raw Images")
# plot_images(fashion_train_images, fashion_train_images_label, "Fashion MNIST Raw Images")

#flatten images
mnist_train_images_flat = mnist_train_images.reshape(mnist_train_images.shape[0], -1)
mnist_test_images_flat = mnist_test_images.reshape(mnist_test_images.shape[0], -1)
fashion_train_images_flat = fashion_train_images.reshape(fashion_train_images.shape[0], -1)
fashion_test_images_flat = fashion_test_images.reshape(fashion_test_images.shape[0], -1)

#before and after flattening 
# print("Original MNIST Train Images Shape: ", mnist_train_images.shape) # (60000, 28, 28) 
# print("Flattened MNIST Train Images Shape:", mnist_train_images_flat.shape) # (60000, 784) 
# print("Original Fashion MNIST Train Images Shape:", fashion_train_images.shape) # (60000, 28, 28) 
# print("Flattened Fashion MNIST Train Images Shape:", fashion_train_images_flat.shape) # (60000, 784) 

#Standardize
scalar = StandardScaler()
mnist_train_images_standardized = scalar.fit_transform(mnist_train_images_flat)
mnist_test_images_standardized = scalar.transform(mnist_test_images_flat)
fashion_train_images_standardized = scalar.fit_transform(fashion_train_images_flat)
fashion_test_images_standardized = scalar.transform(fashion_test_images_flat)

#PCA is sensitive to feature scales. Standardized data should be (mean=0, variance=1) but it's not. Ref: BOOK
# print("MNIST Train Mean:", np.mean(mnist_train_images_standardized)) 
# print("MNIST Train Std:", np.std(mnist_train_images_standardized))    
# print("Fashion MNIST Train Mean:", np.mean(fashion_train_images_standardized))  
# print("Fashion MNIST Train Std:", np.std(fashion_train_images_standardized))    

#Dimensionality Reduction Functions
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
    print(f"\n{dataset_name} PCA (n_components={n_components}) Variance Explained: {var_ratio:.4f}")
    
    plot_pca_variance(pca, dataset_name, n_components)
    return X_train_pca, X_test_pca, var_ratio

def classify_and_predict(X_train, y_train, X_test, y_test, dataset_name, n_components):
    #classifier = LogisticRegression(max_iter=3000)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{dataset_name} Accuracy with {n_components} PCA components: {accuracy:.4f}")
    return accuracy

# for n_components in [50, 100, 200]:
#     apply_pca(mnist_train_images_standardized, mnist_test_images_standardized, n_components, dataset_name="MNIST")
#     apply_pca(fashion_train_images_standardized, fashion_test_images_standardized, n_components, dataset_name="Fashion MNIST")

for n_components in [50, 100, 200]:
    # MNIST
    X_train_mnist_pca, X_test_mnist_pca, _ = apply_pca(
        mnist_train_images_standardized, 
        mnist_test_images_standardized, 
        n_components, 
        dataset_name="MNIST"
    )
    classify_and_predict(X_train_mnist_pca, mnist_train_images_label,
                                  X_test_mnist_pca, mnist_test_images_label,
                                  dataset_name="MNIST", n_components=n_components)
    
    # Fashion MNIST
    X_train_fashion_pca, X_test_fashion_pca, _ = apply_pca(
        fashion_train_images_standardized, 
        fashion_test_images_standardized, 
        n_components, 
        dataset_name="Fashion MNIST"
    )
    classify_and_predict(X_train_fashion_pca, fashion_train_images_label,
                                  X_test_fashion_pca, fashion_test_images_label,
                                  dataset_name="Fashion MNIST", n_components=n_components)


#_____________________________________________________________________________________________________________________ 
# TASK 3.3 
#______________________________________________________________________________________________________________________ 

param_grid_linear = { 
'C': [0.01, 0.1, 1, 10] 
} 

param_grid_rbf = { 
'C': [0.01, 0.1, 1, 10], 
'gamma': [0.0001, 0.001, 0.01, 0.1] 
} 

param_grid_poly = { 
'C': [0.01, 0.1, 1, 10], 
'gamma': [0.0001, 0.001, 0.01, 0.1], 
'degree': [2, 3, 4, 5] 
} 


def train_svc_with_grid_search(X_train, y_train, X_test, y_test, kernel_type, param_grid): 

    print(f"\nRunning Grid Search for kernel = '{kernel_type}'") 

    svc = SVC(kernel=kernel_type) 
    grid_search = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1, verbose=1, scoring='accuracy') 
    start_time = time.time() 
    grid_search.fit(X_train, y_train) 
    elapsed_time = time.time() - start_time 

    print(f"Best parameters for '{kernel_type}': {grid_search.best_params_}") 
    print(f"Training time: {elapsed_time:.2f} seconds") 

    best_model = grid_search.best_estimator_ 
    y_pred = best_model.predict(X_test)

    print(f"Classification Report ({kernel_type}):") 
    print(classification_report(y_test, y_pred)) 

    return best_model, y_pred

def plot_confusion_matrix(y_true, y_pred, title):
    confmat = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title)
    plt.show()

pca_100 = PCA(n_components=100) 
mnist_pca_100_train = pca_100.fit_transform(mnist_train_images_standardized) 
mnist_pca_100_test = pca_100.transform(mnist_test_images_standardized) 

#MNIST compressed data 
X_train = mnist_pca_100_train 
X_test = mnist_pca_100_test 
y_train = mnist_train_images_label 
y_test = mnist_test_images_label 

# Linear kernel 
svc_linear, y_pred_linear = train_svc_with_grid_search(X_train, y_train, X_test, y_test, 'linear', param_grid_linear) 
plot_confusion_matrix(y_test, y_pred_linear, "MNIST PCA(100) - Linear SVC")

# RBF kernel 
svc_rbf, y_pred_rbf = train_svc_with_grid_search(X_train, y_train, X_test, y_test, 'rbf', param_grid_rbf) 
plot_confusion_matrix(y_test, y_pred_rbf, "MNIST PCA(100) - RBF SVC")

# Polynomial kernel 
svc_poly, y_pred_poly = train_svc_with_grid_search(X_train, y_train, X_test, y_test, 'poly', param_grid_poly) 
plot_confusion_matrix(y_test, y_pred_poly, "MNIST PCA(100) - Polynomial SVC")

 
