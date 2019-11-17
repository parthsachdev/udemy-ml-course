# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 17:07:59 2018

@author: sachd
"""

# CLASSIFICATION PROBLEM
# Artificial neural network

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# encoding categorical data
# encoding the independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_country = LabelEncoder()
x[:,1] = labelencoder_x_country.fit_transform(x[:,1])

labelencoder_x_gender = LabelEncoder()
x[:,2] = labelencoder_x_gender.fit_transform(x[:,2])

#only for countries
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
# avoid dummy variable trap
x = x[:, 1:]

#split the data
# cross_validation replaced with 'model_selection'
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# building knn

import keras
from keras.models import Sequential   # initialise neural network
from keras.layers import Dense   # create layers of ANN

# initialising the ANN
# by defining a sequence of layers
classifier = Sequential()

# adding input lyer and the first hidden layer
classifier.add(Dense(output_dim = (1+11)//2, init = 'uniform', activation = 'relu', input_dim = 11))

# adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# add the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# predicting
y_pred = classifier.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)