# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 20:18:17 2020
Content
Columns

age: age of primary beneficiary

sex: insurance contractor gender, female, male

bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9

children: Number of children covered by health insurance / Number of dependents

smoker: Smoking

region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.

charges: Individual medical costs billed by health insurance

@author: rachi
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('insurance.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
X[:, 4] = labelencoder.fit_transform(X[:, 4])
X[:, 5] = labelencoder.fit_transform(X[:, 5])
onehotencoder = OneHotEncoder(categorical_features = [5])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

'''#building an optional model using backward elimination-P-value and adj-RSquared
import statsmodels.api as sm
X = sm.add_constant(X)
X_opt = X[:, [0,4,6,7,8]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''sc_Y = StandardScaler()
y_train = sc_Y.fit_transform(y_train.reshape(-1,1))'''

y_train = y_train.astype('int')

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 5)
#fitting and transforming all the data
X_train = lda.fit_transform(X_train, y_train)
#just transform the data, fitting the data will be done in the final step
X_test = lda.transform(X_test)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_train_pred = regressor.predict(X_train)
y_pred = regressor.predict(X_test)
#y_pred = sc_Y.inverse_transform(y_pred)

#accuracy measurement
'''by using variance score function-If variance score is near about the 1 is perfect prediction'''
print("Variance score for training samples- ",regressor.score(X_train,y_train))
print("Variance score- ",regressor.score(X_test,y_test))

'''by calculaing mean square error'''
print("mean Square error for training samples-",np.mean((y_train_pred-y_train)**2))
print("mean Square error-",np.mean((y_pred-y_test)**2))

'''by calculaing root mean square error'''
print("Root mean Square error for training samples-",np.sqrt(np.mean((y_train_pred-y_train)**2)))
print("Root mean Square error-",np.sqrt(np.mean((y_pred-y_test)**2)))

#outputs
'''use the ColumnTransformer instead.", DeprecationWarning)
Variance score for training samples-  0.716885885281269
Variance score-  0.7775403939785913
mean Square error for training samples- 40588949.355684616
mean Square error- 35399988.72438168
Root mean Square error for training samples- 6370.945719097332
Root mean Square error- 5949.7889646929225'''