# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:07:38 2018

@author: Manish Prasad
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")

X = dataset.iloc[:,:-1].values

y= dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoid Dummy variable trap
X = X[:,1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

# Automated Linear Regression
import statsmodels.formula.api as sm
X = np.append(arr = np.ones([50,1]).astype(int), values = X, axis =1)
#k=[]
def backwardElimination(x,SL) :
    numVars = len(x[0])
    for i in range(0,numVars) :
        regressor_OLS = sm.OLS(y,x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > SL :
            for j in range(0,numVars-i):
                if regressor_OLS.pvalues[j] == maxVar :
                    print('removing',j)
                    x = np.delete(x,j,1)
                    
    regressor_OLS.summary()
    return x
    
SL = 0.05
X_opt = X[:,[0,1,2,3,4,5]]
X_modeled = backwardElimination(X_opt, SL )












