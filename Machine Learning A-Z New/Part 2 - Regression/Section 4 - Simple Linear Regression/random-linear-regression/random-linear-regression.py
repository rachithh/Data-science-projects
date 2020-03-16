# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:57:01 2020

Context
This is probably the dumbest dataset on Kaggle. The whole point is, however, to provide a common dataset for linear regression. Although such a dataset can easily be generated in Excel with random numbers, results would not be comparable.

Content
The training dataset is a CSV file with 700 data pairs (x,y). The x-values are numbers between 0 and 100. The corresponding y-values have been generated using the Excel function NORMINV(RAND(), x, 3). Consequently, the best estimate for y should be x.
The test dataset is a CSV file with 300 data pairs.

@author: rachi
"""

# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset_train = pd.read_csv('train.csv')
dataset_train.fillna(method = 'ffill', inplace = True)
dataset_test = pd.read_csv('test.csv')
X_train = dataset_train.iloc[:, :-1].values
y_train = dataset_train.iloc[:, 1].values
y_train = y_train.round(3)

X_test = dataset_test.iloc[:, :-1].values
y_test = dataset_test.iloc[:, 1].values
y_test = y_test.round(3)

'''# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)'''

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('NORMINV(RAND(), x, 3)- Y vs X (Training set)')
plt.xlabel('NORMINV(RAND(), x, 3)- Y')
plt.ylabel('X (Training set)')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('NORMINV(RAND(), x, 3)- Y vs X (Test set)')
plt.xlabel('NORMINV(RAND(), x, 3)- Y')
plt.ylabel('X (Test set)')
plt.show()