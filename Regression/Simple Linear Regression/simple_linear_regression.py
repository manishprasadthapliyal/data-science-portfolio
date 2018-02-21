# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:59:02 2018

@author: Manish Prasad
"""

# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset containg salary data
dataset = pd.read_csv('Salary_Data.csv')

# matrix of independent variables
X = dataset.iloc[:, :-1].values

#matrix of dependent variables
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# Finding the equation to best fit data.
regressor.fit(X_train, y_train)

# Predicting the Test set results￼
y_pred = regressor.predict(X_test)

# Visualising the Training set results
# scattered plot for all point with Training data
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
# scattered plot for all point with Testing data
plt.scatter(X_test, y_test, color='red')
# we don't need test data because equation is already available both are 
#  going to result same line.
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()













