#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 03:04:15 2018

@author: parth
"""
# Importing Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing Datasets
dataset = pd.read_csv('/home/parth/coding/ML/udemy/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 9 - Random Forest Regression/Random_Forest_Regression/Position_Salaries.csv')
x = dataset.iloc[:, 1:2]
y = dataset.iloc[:, 2]

