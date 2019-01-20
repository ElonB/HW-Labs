#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 7 Feb 2018

@author: elonbrange
"""

#Linear Discriminative Analysis (LDA, Fisher's discriminant analysis)
# Binary class discriminant

#Importing packages
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

#Importing dataset
data0 = np.genfromtxt("0.txt", delimiter = ",")
data1 = np.genfromtxt("1.txt", delimiter = ",")

def disp_img(data, n):
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

#Visualization:  
#zeroes
#a0 = disp_img(data0, 0)
#a1 = disp_img(data0, 1)

#ones
#b0 = disp_img(data1, 0)
#b1 = disp_img(data1, 1)

#Number of classes & features
nclass = 2
features = 784

#Mean vectors Class0 and Class1 
m0 = np.mean(data0, 0)
m1 = np.mean(data1, 0)

mv = []
mv.append(m0)
mv.append(m1)

#Within-class scatter matrix
#Calculating within-scatter for each class separately
#Class0
Sw0 = np.zeros((784, 784))
for i in range(1, len(data0)):
    row = data0[i,:].reshape(784,1)
    mvec = m0.reshape(784, 1)
    Sw0 += (row - mvec).dot((row - mvec).T)

#Class1
Sw1 = np.zeros((784, 784))
for i in range(1, len(data1)):
    row = data1[i,:].reshape(784,1)
    mvec = m1.reshape(784, 1)
    Sw1 += (row - mvec).dot((row - mvec).T)

#Within-scatter matrix    
Sw = Sw0 + Sw1

#Alternative way, calculating covariance
cov0 = np.cov(data0,rowvar = False)
cov1 = np.cov(data1, rowvar = False)
S0 = 234*cov0
S1 = 277*cov1

covSw = S0 + S1

#Cross-checking methods
if covSw.all() == Sw.all():
    print("Within-class scatter OK")

#Between-class scatter matrix
#Calculating overall mean for each feature
data = np.concatenate((data0, data1), axis = 0)
mOverall = np.mean(data, 0)

#mOverall = np.mean(mv)
mOverall = mOverall.reshape(784, 1)

#B/w-scatter matrix class0
mv0 = m0.reshape(784, 1)
Sb0 = (mv0 - mOverall).dot((mv0 - mOverall).T)

#B/w-scatter matrix class0
mv1 = m1.reshape(784,1)
Sb1 = (mv1 - mOverall).dot((mv1 - mOverall).T)

#Between-scatter matrix
Sb = 234*Sb0 + 277*Sb1

#Sb = (mv1 - mv0).dot((mv1 - mv0).T)

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
W = np.hstack((eigPair[0][1].reshape(784,1)))

#Transforming data samples to new sub-space
lda0 = data0.dot(W)
lda1 = data1.dot(W)

#Decision boundary
bl = 0.5*(np.mean(lda1) - np.mean(lda0))

#Plotting histogram of classes
plt.hist(lda0, color = "blue")
plt.hist(lda1, color = "orange")
plt.title("Histogram of classes")
plt.show()

#Plotting the samples against gaussian noise
noise0 = np.random.normal(0, 1, 234)
noise1 = np.random.normal(0, 1, 277)

plt.scatter(lda0, noise0, marker = "x", color = "blue")
plt.scatter(lda1, noise1, marker = "x", color = "orange")
plt.legend([0,1], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('LDA, W transform')
plt.ylabel('Gaussian noise')
plt.show()

#Alternative plot
plt.scatter(lda0, noise0, color = "white")
plt.scatter(lda1, noise1, color = "white")
for x, y in zip(lda0, noise0):
    plt.text(x, y, str(0), color="blue", fontsize=10)
    
for x, y in zip(lda1, noise1):
    plt.text(x, y, str(1), color="orange", fontsize=10)

#Decision boundary, 
bl = 0.5*(np.mean(lda1) - np.mean(lda0))
tbl = np.mean(np.concatenate((lda0, lda1),axis = 0))
#Alternative calculation (orojecting the means)
#bl = 0.5*(m1.dot(W) - m0.dot(W)) 

x = [tbl, tbl]
y = [min((min(noise0), min(noise1))), max((max(noise0), max(noise1)))]
plt.plot(x, y)

#plt.legend([0,1], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('LDA, W transform')
plt.ylabel('Gaussian noise')
plt.show()



