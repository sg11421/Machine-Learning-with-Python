#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 22:21:27 2024

@author: anishananda
"""
'''
Dataset: Red Wine Quality

The dataset is related to the red variant of "Vinho Verde" wine. It contains 
1599 data points where features are the physicochemical properties and the 
target value is quality which is an integer score ranging from 0-10. Your task 
is to classify if the wine provided is good based on its physicochemical properties.

(i) Create a new column on the dataset with binary values (i.e, 0 or 1) telling 
whether the wine is of good quality or not. You can categorise wines with 
quality>=7 to be of good quality. Drop the original ‘quality’ column.

(ii) Perform the data pre-processing steps that you feel are important for the 
given dataset.

(iii) Apply following classification algorithms on the given dataset (you are 
allowed to use scikit-learn library until not specified ‘from scratch’):

 a) Logistic Regression
 b) K-Nearest Neighbors
 c) Decision Trees Classifier
 d) Random Forest Classifier
 e) Logistic Regression from Scratch 

(iv) Evaluate all your models based on the accuracy score and f1 score obtained 
on the test dataset.
'''

# libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# data loading
data = pd.read_csv("winequality-red.csv")
data['binary_quality'] = 0


# (i)

data['binary_quality'] = (data['quality'] >= 7).astype(int)

data.drop('quality', inplace=True, axis=1)


# (ii)

data.dropna(inplace = True)

data.drop_duplicates(inplace=True)

X = data.drop(columns = ['binary_quality'])
y = data['binary_quality']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=15)

scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)


# (iii)

# (a) Logistic Regression

logreg_model = LogisticRegression()

logreg_model.fit(Xtrain, ytrain)

ypred = logreg_model.predict(Xtest)


# evaluating (iv)
accuracy = accuracy_score(ytest, ypred)
conf_matrix = confusion_matrix(ytest, ypred)
class_report = classification_report(ytest, ypred)

print("\t\t*** Logistic Regression (using scikit) ***\n")
print("Accuracy: ", accuracy*100, "%")
print("\nConfusion Matrix:\n")
print(conf_matrix)
print("\nClassification Report:\n")
print(class_report)


# (b) K-nearest neighbours

knn_model = KNeighborsClassifier(n_neighbors=2)

knn_model.fit(Xtrain, ytrain)

ypred = knn_model.predict(Xtest)


# evaluating (iv)
accuracy = accuracy_score(ytest, ypred)
conf_matrix = confusion_matrix(ytest, ypred)
class_report = classification_report(ytest, ypred)

print("\t\t*** K nearest neighbours ***\n")
print("Accuracy: ", accuracy*100, "%")
print("\nConfusion Matrix:\n")
print(conf_matrix)
print("\nClassification Report:\n")
print(class_report)


# (c) Decision Tree Classifier

dt_model = DecisionTreeClassifier()

dt_model.fit(Xtrain, ytrain)

ypred = dt_model.predict(Xtest)


# evaluating (iv)
accuracy = accuracy_score(ytest, ypred)
conf_matrix = confusion_matrix(ytest, ypred)
class_report = classification_report(ytest, ypred)

print("\t\t*** Decision Tree Classifier ***\n")
print("Accuracy: ", accuracy*100, "%")
print("\nConfusion Matrix:\n")
print(conf_matrix)
print("\nClassification Report:\n")
print(class_report)


# (d) Random forest classifier

rf_model = RandomForestClassifier()

rf_model.fit(Xtrain, ytrain)

ypred = rf_model.predict(Xtest)


# evaluating (iv)
accuracy = accuracy_score(ytest, ypred)
conf_matrix = confusion_matrix(ytest, ypred)
class_report = classification_report(ytest, ypred)

print("\t\t*** Random Forest Classifier ***\n")
print("Accuracy: ", accuracy*100, "%")
print("\nConfusion Matrix:\n")
print(conf_matrix)
print("\nClassification Report:\n")
print(class_report)


# (e) Logistic Regression from scratch

def sigmoid(z):
  return 1/(1 + np.exp(-z))

def Logistic_Reg(X,y,lr,lamda_reg):
    n,d = X.shape
    X = np.array(X)
    X = (X - X.mean())/X.std()
    X = np.c_[np.ones((n, 1)), X]
    y = np.array(y)
    w=np.zeros(d+1)
    for i in range(n):
        h = sigmoid(np.dot(X[i],w))
        gradient=(np.dot(X.T,(h-y)) + lamda_reg*w)/(n)
        w=w-lr*gradient
    return w

def Logistic_pred(X, w):
    m,n = X.shape
    X = np.array(X)
    X = (X - X.mean())/X.std()
    X = np.c_[np.ones((m, 1)), X]
    ypred = sigmoid(np.dot(X, w))
    ypred = (ypred>0.5).astype(int)
    return ypred


lr=0.0001
lamda_reg = 0.5
w=Logistic_Reg(Xtrain,ytrain,lr,lamda_reg)
ypred = Logistic_pred(Xtest, w)
print(ypred)


# evaluating (iv)
accuracy = accuracy_score(ytest, ypred)
conf_matrix = confusion_matrix(ytest, ypred)
class_report = classification_report(ytest, ypred)

print("\t\t*** Logistic Regression from Scratch ***\n")
print("Accuracy: ", accuracy*100, "%")
print("\nConfusion Matrix:\n")
print(conf_matrix)
print("\nClassification Report:\n")
print(class_report)

