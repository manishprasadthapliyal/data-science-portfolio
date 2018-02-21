# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:54:16 2018

@author: Manish Prasad
"""
# Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Linear Regression to the dataset.
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting Polynomial Regression to the dataset.
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) 
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# Visualising the Linear Regression Results
plt.scatter(X,y,color = 'red')
plt.plot(X, lin_reg.predict(X),color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel = "Position Level"
plt.xlabel = "Salary"
plt.show()

# Visualising the Polynomial Regression Results
x_grid = np.arange(min(X),max(X),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(X,y,color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel = "Position Level"
plt.xlabel = "Salary"
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))














# Visualising the Polynomial Regression Results