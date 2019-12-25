# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:46:33 2019

@author: Guru  Prasad Muppana
"""

# Sign Language digits.

#Details of datasets:
#Image size: 64x64
#Color space: Grayscale
#File format: npy
#Number of classes: 10 (Digits: 0-9)
#Number of participant students: 218
#Number of samples per student: 10

# Opening .npy files .
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics




import os
from os import listdir


from PIL import Image as pil

from scipy.misc import imread, imresize
from keras.utils import to_categorical

#from sklearn.model_selection import train_test_split



# calculate the cross-entropy error

#def cross_entropy(T, Y):
#    E = 0
#    for i in range(len(T)):
#        if T[i] == 1:
#            E -= np.log(Y[i])
#        else:
#            E -= np.log(1 - Y[i])
#    return E

def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# make predictions
def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W, b):
    return softmax(X.dot(W) + b)

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)

# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY))

# Settings:

img_size = 64
grayscale_images = True
num_class = 10
test_size = 0.2

dataset_path = "D:\python\Guru\Datasets\signlanguagedigits"


#img=mpimg.imread(X_train[0])
#plt.imshow(X_train[0].reshape(64,64),cmap="gray")
lable_map= ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]
argmax_num = [9,0,7,6,1,8,4,3,2,5]


def get_img(data_path):
    # Getting image array from path:
    img = imread(data_path, flatten=grayscale_images)
    img = imresize(img, (img_size, img_size, 1 if grayscale_images else 3))
    return img


def swap(Xt,a,b):
    N,_ = Xt.shape
    temp = np.random.randn(N,1)
    temp[:,0] = Xt[:,a]
    Xt[:,a] = Xt[:,b]
    Xt[:,b]=temp[:,0]


#temp[:,0] = Y[:,0]
#Y[:,0] = Y[:,1]
#Y[:,1]=temp[:,0]

#The files can be downloaded from Kaggle    
# https://www.kaggle.com/datamunge/sign-language-mnist
    # The files are numpy files
    
X = np.load("X.npy")
Y = np.load("Y.npy")

lable_map= ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]
argmax_num = [9,0,7,6,1,8,4,3,2,5]

b=2
swap(Y,0,1) # [9,0,7,6,1,8,4,3,2,5] ->[0,9,7,6,1,8,4,3,2,5]
swap(Y,1,4) # [0,9,7,6,1,8,4,3,2,5] ->[0,1,7,6,9,8,4,3,2,5]
swap(Y,2,8) # [0,1,7,6,9,8,4,3,2,5] ->[0,1,2,6,9,8,4,3,7,5]
swap(Y,3,7) # [0,1,2,6,9,8,4,3,7,5] ->[0,1,2,3,9,8,4,6,7,5]
swap(Y,4,6) # [0,1,2,3,9,8,4,6,7,5] ->[0,1,2,3,4,8,9,6,7,5]
swap(Y,5,9) # [0,1,2,3,4,8,9,6,7,5] ->[0,1,2,3,4,5,9,6,7,8] 
swap(Y,6,7) # [0,1,2,3,4,5,9,6,7,8] ->[0,1,2,3,4,5,6,9,7,8]
swap(Y,7,8) # [0,1,2,3,4,5,6,9,7,8] ->[0,1,2,3,4,5,6,7,9,8]
swap(Y,8,9) # [0,1,2,3,4,5,6,7,9,8] ->[0,1,2,3,4,5,6,7,8,9]

## swaping 9 and 0
##temp =np.array((2062,10))
#temp = np.random.randn(2062,1)
#temp[:,0] = Y[:,0]
#Y[:,0] = Y[:,1]
#Y[:,1]=temp[:,0]
## [0,9,7,6,1,8,4,3,2,5]
#temp[:,0] = Y[:,1]
#Y[:,1] = Y[:,4]
#Y[:,4]=temp[:,0]
## [0,1,7,6,9,8,4,3,2,5]
#temp[:,0] = Y[:,2]
#Y[:,2] = Y[:,8]
#Y[:,8]=temp[:,0]
## [0,1,2,6,9,8,4,3,7,5]
#temp[:,0] = Y[:,3]
#Y[:,3] = Y[:,7]
#Y[:,7]=temp[:,0]
## [0,1,2,3,9,8,4,6,7,5]
#temp[:,0] = Y[:,4]
#Y[:,4] = Y[:,6]
#Y[:,6]=temp[:,0]
## [0,1,2,3,4,8,9,6,7,5]
#temp[:,0] = Y[:,5]
#Y[:,5] = Y[:,9]
#Y[:,9]=temp[:,0]
## [0,1,2,3,4,5,9,6,7,8]
#temp[:,0] = Y[:,6]
#Y[:,6] = Y[:,7]
#Y[:,7]=temp[:,0]
## [0,1,2,3,4,5,6,9,7,8]
#temp[:,0] = Y[:,7]
#Y[:,7] = Y[:,8]
#Y[:,8]=temp[:,0]
## [0,1,2,3,4,5,6,7,9,8]
#temp[:,0] = Y[:,8]
#Y[:,8] = Y[:,9]
#Y[:,9]=temp[:,0]
#

#temp= Y[1,4,8,7,9,3,2,5,0]
#Y = temp

a=1

# X -> len 2062 and Y len 2062. There are 2062 samples for both X and Y.
# diviving the X and Y into two sets.
# X training data and Y is evaluation data.
# Not sure what random_state means . it is assigned as 42.

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

# concatenage
C = pd.DataFrame()
#C = pd.concat(X,Y,axis=1)

X = X.reshape(2062,64*64)
C  = np.concatenate((X,Y),axis=1)
np.random.shuffle(C)
# split X and Y
X = C[:,:64*64]
Y = C[:,64*64:]

## Mean images:
#Y_argmax = np.argmax(Y, axis=1)
## loop through each label
#for k in range(10):
#  Xk = X[Y_argmax == k]
#
#  # mean image
#  Mk = Xk.mean(axis=0)
#
#  # reshape into an image
#  im = Mk.reshape(64, 64)
#
#  # plot the image
#  plt.imshow(im, cmap='gray')
#  plt.title("Label: %s" % k)
#  plt.show()
## Mean images:



X = np.vstack((X,X))
X = np.vstack((X,X))

Y = np.vstack((Y,Y))
Y = np.vstack((Y,Y))

N, D = X.shape

print("Shape: ", N, D)
print("\n")

T = np.argmax(Y, axis=1)

train_img, test_img, train_lbl, test_lbl = train_test_split(X,T,test_size=.25,random_state=42)


scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_img)
# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

#model = LogisticRegression(solver = 'lbfgs')
model = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

print("Training the model began")


model.max_iter = 200
model.verbose = 1
 
print("max iter:\n", model.max_iter)

model.fit(train_img, train_lbl)

print("Training successfull")
# use the model to make predictions with the test data
y_pred = model.predict(test_img)
# how did our model perform?
count_misclassified = (test_lbl != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(test_lbl, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


t= test_img[test_lbl != y_pred]

#test_lbl.reshape((1, -1))

test_lbl = np.array(test_lbl) # Key conversion

lbl = test_lbl[(test_lbl != y_pred)]
pred = y_pred[(test_lbl != y_pred)]

#import matplotlib.pyplot as plt



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_lbl,y_pred)
print ("Confusion Matrics:\n", cm)

rows = int( count_misclassified/5 + 1)
columns = 5

fig = plt.figure()
image_size = 64
for i in range(rows):
    for j in range(columns):
        if i*5+j >= count_misclassified:
            print("count:",i*5+j)
            break
        a = fig.add_subplot(rows, 5, i*5+j+1)
        img = np.array(t[i*5+j]);
        #print(i*5+j)
        img = img.reshape(image_size,image_size)
        #image =np.asarray(images[0]).squeeze()
        imgplot = plt.imshow(img)
        #imgplot.set_clim(0.0, 0.7)
        a.set_title(str(lbl[i*5+j])+","+str(pred[i*5+j]))

plt.show()

test_file9 = "D:\python\Guru\Datasets\signlanguagedigits\Test\pic_9_64x64.jpg"


#
#def rgb2gray(rgb):
#    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
#
#
#
#img = mpimg.imread(test_file)     
#gray = rgb2gray(img)    
#plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
#plt.show()

from skimage import io


img = io.imread(test_file9, as_gray=True)

imgplot = plt.imshow(img)
plt.show()
#img1 = img.reshape(64*64)
img1 = img.reshape(1,-1)
img1 = scaler.transform(img1)
number = model.predict(img1)
print("Orginal 6 and actual", number)

# Added a new comments at the end using the GitHub editor

# This is a new commment added from Spyer app. Now, plans to move this with GitHub Desktop app

