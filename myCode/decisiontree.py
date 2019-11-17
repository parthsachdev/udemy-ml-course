#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 23:40:20 2018

@author: parth
"""
# Decision Tree Regression


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Datasets
dataset = pd.read_csv('/home/parth/coding/ML/udemy/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/SVR/Position_Salaries.csv')
x = dataset.iloc[:, 1:2]
y = dataset.iloc[:, 2:]

# NO FEATURE SCALING

# Regressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising Decision Tree
# Higher Resolution
X_grid = np.arange(min(x['Level']), max(x['Level']), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.show()