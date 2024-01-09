#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 23:17:07 2023

@author: anishananda
"""

'''
Dataset: Air quality of an Italian city

The dataset contains 9358 instances of hourly averaged responses from an array
of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multi 
Sensor Device. The device was located on the field in a significantly polluted 
area, at road level, within an Italian city. Data were recorded from March 2004 
to February 2005 (one year) representing the longest freely available 
recordings of on field deployed air quality chemical sensor devices responses. 
Ground Truth hourly averaged concentrations for CO, Non Metanic Hydrocarbons, 
Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2) and were 
provided by a co-located reference certified analyzer. Missing values are 
tagged with -200 values.

Your objective is to predict the Relative Humidity of a given point of time 
based on all other attributes affecting the change in RH.

(i) Perform the data pre-processing steps on the dataset as explained in the 
class. Handle missing values, get insights from correlation matrix and deal 
with outliers.

(ii) Split the dataset into a 85:15 ratio into training and test dataset using 
the sklearn library.

(iii) Train a linear regression model from scratch using only numpy, pandas and
matplotlib and train a linear regression model using the sklearn library on the 
training dataset.

(iv) Calculate the r2 score and mean squared error using the test dataset. 
Compare the results obtained and plot your results.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error




# (i)

# loading the dataset
data = pd.read_csv("AirQualityUCI.csv") 
#X = data.iloc[:, 1:13] + data.iloc[:, 14]
#y = data.loc[:, "RH"]


# preprocessing

# we drop rows which give us very little information
data.dropna(axis=1, how='all', inplace=True)
data.replace(-200, np.nan, inplace=True)
#data.dropna(thresh=len(data)*0.7, axis=1, inplace=True)

# and replace others with mean of column
data.fillna(data.mean(numeric_only = True), inplace=True)



 # insights from correlation matrix
correlation_matrix = data.drop(columns = ['Date', 'Time']).corr()


plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='viridis', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.show()

for cn in data.drop(columns = ['Date', 'Time']) :
    m = data[cn].mean()
    stdev = data[cn].std()
    lb = m - 5*stdev
    ub = m + 5*stdev
    # find and remove outliers
    outliers = (data[cn] < lb) | (data[cn] > ub)
    data_no_outliers = data[~outliers]
    # visualizing the distribution before and after removing outliers
    plt.figure(figsize=(12, 6))
    #before
    plt.subplot(1, 2, 1)
    plt.boxplot(data[cn])
    plt.title('Before Removing Outliers')
    plt.xticks([1], [cn])
    #after
    plt.subplot(1, 2, 2)
    plt.boxplot(data_no_outliers[cn])
    plt.title('After Removing Outliers')
    plt.xticks([1], [cn])
    
    plt.show()
    data = data_no_outliers
    

X = data.drop(columns = ["Date", "Time", "RH"])
y = data["RH"]




# (ii)

# train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size = 0.15, random_state = 50)



# (iii)

# linear regression from scratch

def LR( X, y, lr, lamdareg):
    m,n = X.shape
    y = np.array(y)
    # normalising
    Xp = (X - X.mean(axis=0))/X.std(axis=0)
    # adding a column of ones for bias term
    Xp = np.c_[np.ones((m, 1)), Xp]
    # initialising weight vector
    w = np.random.randn(n+1)
    # training
    for i in range(m):
        g = np.dot(Xp[i],w)
        gradient = ((g - y[i])*Xp[i] + lamdareg*w)/m
        w=w-lr*gradient
    return w


def LR_predict(w, X):
    m,n = X.shape
    Xp = (X - X.mean())/X.std()
    Xp = np.c_[np.ones((m, 1)), Xp]
    ypred = np.dot(Xp, w)
    return ypred


w = LR(Xtrain,ytrain, 25, 0.0001)
ypred_scratch = LR_predict(w, Xtest)




# using SKLearn

# training linear regression model using sklearn
linear_reg_sklearn = LinearRegression()
linear_reg_sklearn.fit(Xtrain, ytrain)


# predicting on test set
ypred_sklearn = linear_reg_sklearn.predict(Xtest)




# (iv)

# comparing

r2_scratch = r2_score(ytest, ypred_scratch)
mse_scratch = mean_squared_error(ytest, ypred_scratch)

r2_sklearn = r2_score(ytest, ypred_sklearn)
mse_sklearn = mean_squared_error(ytest, ypred_sklearn)

print("Results from Scratch:")
print("R2 Score:", r2_scratch)
print("Mean Squared Error:", mse_scratch)
print()
print("Results from sklearn:")
print("R2 Score:", r2_sklearn)
print("Mean Squared Error:", mse_sklearn)

# plotting 

plt.scatter(ytest, ypred_scratch, label='Scratch')
plt.scatter(ytest, ypred_sklearn, label='SKLearn')
plt.xlabel('Actual RH')
plt.ylabel('Predicted RH')
plt.legend()
plt.title('Comparison of Linear Regression Models')
plt.show()




