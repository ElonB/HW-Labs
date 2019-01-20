#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 17:01:50 2018

@author: elonbrange
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 7 Feb 2018

@author: elonbrange
"""

#Linear Discriminative Analysis (LDA, Fisher's discriminant analysis)

#Importing packages
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

#Importing dataset
data = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")

#X = data[0:2000]
#y = labels[0:2000]

X = data
y = labels

test_X = data[2001: 2500]
test_y = labels[2001: 2500]



def disp_img(data, n):
    plt.clf()
    y = np.empty((28, 28))
    for i in range(1, 27):
        low = 28*(i-1)
        up = 28*i -1
        for j in range(0, 27):
            y[i,j] = data[n, low:up][j]
    y = np.transpose(y)
    plt.matshow(y)
    plt.show
    return y

##Number of classes & features
nclass = 10

##Mean vectors 
#m0 = np.mean(data0, 0)
mv = []
tmp = []
for i in range(0, nclass):
    for j in range(len(X)):
        if i == y[j]:
            tmp.append([X[j]])
    mv.append(np.mean(tmp, 0))
    tmp = []

#Within-class scatter matrix
#Calculating within-scatter for each class separately
SwC = []
tmp = []
for i in range(0, nclass):
    SwTemp = np.zeros((784, 784))
    c = 0
    for j in range(len(X)):
        if i == y[j]:
            #tmp.append(X[j])
            r = X[j]
            tmp = np.append(tmp, r)
            c = c + 1
    tmp = tmp.reshape(c, 784)
    for k in range(1, len(tmp)):
        row = tmp[k,:].reshape(784,1)
        mvec = mv[i]
        mvec.reshape(784, 1)
        SwTemp += (row - mvec).dot((row - mvec).T)
    SwC.append(SwTemp)
    tmp = []
    
#Within-scatter matrix    
Sw = sum(SwC)

##Between-class scatter matrix
##Calculating overall mean for each feature
mOverall = np.mean(X, 0)
mOverall = mOverall.reshape(784, 1)

#No of samples from each class
cl = []
for i in range(0, nclass):
    c = 0
    for j in range(len(X)):
        if i == y[j]:
            c = c + 1
    cl = np.append(cl, c)
        
    
#B/w-scatter matrix
SbC = []
Sb = np.zeros((784,784))
for i in range(0, nclass):
    c = 0
    for j in range(len(X)):
        if i == y[j]:
            c = c + 1
    mvT = mv[i]
    mvT = mvT.reshape(784, 1)
    SbC = ((mvT - mOverall).dot((mvT - mOverall).T))
    
    Sb += Sb + c*SbC

#Solving eigenvalue problem
#Calculating (Moore-Penrose) pseudo inverse since Sw is singluar ->
SwInv = linalg.pinv(Sw)

#Eig to sove
eigVal, eigVec = linalg.eig(SwInv.dot(Sb))

#Eigenvalues and Eigenvectors tuples/pairs
eigPair = [(np.abs(eigVal[i]), eigVec[:,i]) for i in range(len(eigVal))]

#Sorting Eigenvalues in descending order
eigPair = sorted(eigPair, key=lambda k: k[0], reverse = True)

#Explanation of variance
varExpl = []
eigSum = sum(eigVal)
for i,j in enumerate(eigPair):
    varExpl.append('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigSum).real))

#Choosing eigenvectors with largest eigenvalues
#W = np.hstack((eigPair[0][1].reshape(784,1)))
W = np.hstack((eigPair[0][1].reshape(784,1), eigPair[1][1].reshape(784,1)))

##Transforming data samples to new sub-space
ldaX = X.dot(W)

#plt.scatter(ldaX[0,:], ldaX[1,:])
color = ['blue', 'red', 'green', 'yellow', 'orange', 'teal', 'cyan', 'pink', 'gray', 'black']
for i in range(0, nclass):
    for j in range(len(X)):
        if i == y[j]:
            plt.scatter(ldaX[j,0], ldaX[j,1], color = color[i]) #np.random.normal(0, 1)
plt.show()


##Decision boundary
#bl = 0.5*(np.mean(lda1) - np.mean(lda0))
#
##Plotting histogram of classes
#plt.hist(lda0, color = "blue")
#plt.hist(lda1, color = "orange")
#plt.title("Histogram of classes")
#plt.show()
#
##Plotting the samples against gaussian noise
#noise0 = np.random.normal(0, 1, 234)
#noise1 = np.random.normal(0, 1, 277)
#
#plt.scatter(lda0, noise0, marker = "x", color = "blue")
#plt.scatter(lda1, noise1, marker = "x", color = "orange")
#plt.legend([0,1], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.xlabel('LDA, W transform')
#plt.ylabel('Gaussian noise')
#plt.show()
#
##Alternative plot
#plt.scatter(lda0, noise0, color = "white")
#plt.scatter(lda1, noise1, color = "white")
#for x, y in zip(lda0, noise0):
#    plt.text(x, y, str(0), color="blue", fontsize=10)
#    
#for x, y in zip(lda1, noise1):
#    plt.text(x, y, str(1), color="orange", fontsize=10)
#
##Decision boundary, 
#bl = 0.5*(np.mean(lda1) - np.mean(lda0))
#tbl = np.mean(np.concatenate((lda0, lda1),axis = 0))
##Alternative calculation (orojecting the means)
##bl = 0.5*(m1.dot(W) - m0.dot(W)) 
#
#x = [tbl, tbl]
#y = [min((min(noise0), min(noise1))), max((max(noise0), max(noise1)))]
#plt.plot(x, y)
#
##plt.legend([0,1], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.xlabel('LDA, W transform')
#plt.ylabel('Gaussian noise')
#plt.show()
#
#
#
