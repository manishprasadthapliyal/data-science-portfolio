"""
Multiple Linear Regression

"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
# Assigning all the Independent variable to X
X = dataset.iloc[:, :-1].values
# Assigning the dependent variable to y
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy variable trap. remove atleast one dummy variable.
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set. assigning 20% for testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
# Creating Regressor object. In other word a machine
regressor = LinearRegression()
# We will train our model. In other word learning
regressor.fit(X_train,y_train)

# Predict Test Data
y_pred = regressor.predict(X_test)

# Building optimal solution using Backward Elimination
# Importing Statsmodels.formula.api
import statsmodels.formula.api as sm

# y= b0 + b1x+ ... bNxn . we add our constant by adding a coulmn of 1's
# np.ones returns a column of 1's followed by X.
X = np.append(arr = np.ones([50,1]).astype(int), values = X, axis =1)

# We will now try to create optimal model by Backward Elimination.
# We will take significance level = 0.05  i.e 5%.
X_opt = X[:,[0,1,2,3,4,5]]

# We  will fit the model with all predictors
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

# Consider predictor with highest P-value.
regressor_OLS.summary()

# if P-value > SL. Remove the predictor. 
X_opt = X[:,[0,1,3,4,5]]

# Fit the model without the removed variable.
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

# Consider predictor with highest P-value.
regressor_OLS.summary()

# if P-value > SL. Remove the predictor. 
X_opt = X[:,[0,3,4,5]]

# Fit the model without the removed variable.
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

# Consider predictor with highest P-value.
regressor_OLS.summary()

# if P-value > SL. Remove the predictor. 
X_opt = X[:,[0,3,5]]

# Fit the model without the removed variable.
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

# Consider predictor with highest P-value.
regressor_OLS.summary()

# if P-value > SL. Remove the predictor. 
X_opt = X[:,[0,3]]
# if P-value < SL. Finish 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# You have your optimal model.
regressor_OLS.summary()





















