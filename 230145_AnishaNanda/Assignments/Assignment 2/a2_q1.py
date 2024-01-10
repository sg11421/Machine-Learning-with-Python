#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 14:12:15 2023

@author: anishananda
"""

'''
Generate a dataset using numpy random as I had shown in the 1st class of this week. 
Train a linear regression model  using |x-xhat|^3 as your loss function and 
a polynomial regression model using |x-xhat|^7 as your loss function. 
(Note that you will need to derive the gradient descent algorithms for these functions yourselves). 
You are allowed to use only numpy, pandas and Matplotlib. 
Then train a linear regression model using the sklearn library on the same dataset. 
At last, plot the dataset and curves obtained from all models in the same figure.
'''

import numpy as np
import matplotlib.pyplot as plt


# generating a dataset
np.random.seed(10)
rng = np.random.RandomState(10)
Xtrain = np.random.randint(1,100, size=(200,1))
Xtest = np.random.randint(1,100, size=(50,1))
ytrain = 3*Xtrain.flatten() + 20*rng.randn(200)
ytest = 3*Xtest.flatten() + 20*rng.randn(50)
plt.scatter(Xtrain, ytrain, color = 'blue', label = 'training data')
plt.scatter(Xtest, ytest, color= 'green', label = 'testing data')
plt.legend()
plt.show()


# loss function = [x - xhat]^3
# y = w0 + w1*x1
# up.date rule: (w= w-lr*gradient)
    # h(k+1)= h(k) + lr*3*(x-xhat)^2


def LR( X, y, lr, lamdareg):
    m,n = X.shape
    # normalising
    Xp = (X - X.mean(axis=0))/X.std(axis=0)
    # adding a column of ones for bias term
    Xp = np.c_[np.ones((m, 1)), Xp]
    # initialising weight vector
    w = np.random.randn(n+1)
    # training
    for i in range(m):
        g = np.dot(Xp[i],w)
        #gradient = ((np.dot(Xp.T,-3*((g-y)**2))) + w*lamdareg)/m
        gradient = (-3*((g - y[i])** 2) * Xp[i] + lamdareg*w)/m
        w=w-lr*gradient
    return w


def CL_LR_predict(w, X):
    m,n = X.shape
    Xp = (X - X.mean())/X.std()
    Xp = np.c_[np.ones((m, 1)), Xp]
    ypred = np.dot(Xp, w)
    return ypred


w = LR(Xtrain,ytrain, 0.01, 0.1)
ypred_CL_LR = CL_LR_predict(w, Xtest)


# Plotting the results for CL LR
plt.scatter(Xtrain, ytrain, color='black', label='data')
plt.plot(Xtest, ypred_CL_LR, color='red', label='prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression using Custom Loss')
plt.legend()
plt.show()




def LR_poly( X, y, lr, lamdareg):
    m,n = X.shape
    # normalising
    Xp = (X - X.mean(axis=0))/X.std(axis=0)
    # adding a column of ones for bias term
    Xp = np.c_[np.ones((m, 1)), Xp]
    # initialising weight vector
    w = np.random.randn(n+1)
    # training
    for i in range(m):
        g = np.dot(Xp[i],w)
        #gradient = ((np.dot(Xp.T,-3*((g-y)**2))) + w*lamdareg)/m
        gradient = (-7*((g - y[i])** 6) * Xp[i] + lamdareg*w)/m
        w=w-lr*gradient
    return w


w = LR(Xtrain,ytrain, 0.01, 0.1)
ypred_CL_LR_poly = CL_LR_predict(w, Xtest)


# Plot the results
plt.scatter(Xtrain, ytrain, color='black', label='data')
plt.plot(Xtest, ypred_CL_LR_poly, color='blue', label='Polynomial Regression using Custom Loss')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression with Custom Loss')
plt.legend()
plt.show()




def LR_sklearn(Xtrain, ytrain, Xtest):
    from sklearn.linear_model import LinearRegression
    # Training linear regression model with sklearn
    linear_reg = LinearRegression()
    linear_reg.fit(Xtrain, ytrain)
    # Predicting using the sklearn model
    ypred = linear_reg.predict(Xtest)
    return ypred


# Plotting the results for sklearn LR
ypred_sklearn = LR_sklearn(Xtrain, ytrain, Xtest)
plt.scatter(Xtrain, ytrain, color='black', label='data')
plt.plot(Xtest, ypred_sklearn, color='green', label='prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression using sklearn')
plt.legend()
plt.show()





# Plotting all models in the same figure
plt.scatter(Xtrain, ytrain, color='black', label='Data')
plt.plot(Xtest, ypred_CL_LR, color='red', label='Linear Regression with Custom Loss')
plt.plot(Xtest, ypred_CL_LR_poly, color='blue', label='Polynomial Regression with Custom Loss')
plt.plot(Xtest, ypred_sklearn, color='green', label='Linear Regression with sklearn')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Comparing of Regression Models')
plt.legend()
plt.show()
