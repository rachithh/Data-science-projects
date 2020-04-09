# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:34:03 2020
Dataset Overview: The data contains information on withdrawals and deposits for a large bank in the Middle East. The original dataset has information for 1100 branches spread over 15 years, however for the sake of brevity we are restricting ourselves to daily aggregated data for just 4 branches over 1 year. 
 
Objective: For small transactions (typically $1000 or less), people go withdraw money from ATMs. For large transactions (say $20,000 or so), people need to visit their branch and withdraw money from there. Consequently, every branch keeps millions of dollars in their vault at any given time to meet customer requirements. 
 
The problem with keeping so much extra money is that a large amount of this cash lies in the vault, totally unused, and thus doesn’t earning interest. If the bank knew exactly how much money would be withdrawn in any given day for each branch, they would be able to keep just the right amounts of cash and earn interest by lending out the remaining cash which is no longer needed. Optimizing money this way results in huge annual savings for the bank, while still meeting customer requirements. 
 
Your job is to predict daily cash withdrawals per branch (“Branch Cash Withdrawal” in the dataset), using any machine learning framework (language / statistical model / library) of your choice. Optimize your models to obtain the lowest possible RMSE.

Note:Since this is a bank in the Middle East, the weekend falls on Friday / Saturday. All branches are closed on weekends. 
@author: rachi
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

# Importing the dataset
dataset = pd.read_csv('branch_transactions_small.csv')

# Abu Nusier branch dataset for branch cash withdrawal
dataset_abu_nusier = dataset[(dataset['Branch.Name.In.English'].str.rstrip()=='Abu Nusier') & (dataset['Activity.Type'].str.rstrip() == 'Branch Cash Withdrawal')]

# select only date and amount withdrawn column for arima
df = dataset_abu_nusier.iloc[:,[0,6]]
df.columns = ["Date","Amount_withdrawn"] #renaming the columns

#change column datatype and also remove negative sign
df['Amount_withdrawn'] = df['Amount_withdrawn'].str.replace(',','').str.replace('-','')
df['Amount_withdrawn'] = df['Amount_withdrawn'].astype(float)

'''data contains outliers'''
df = df[df['Amount_withdrawn']<df['Amount_withdrawn'].quantile(0.987)]

# Convert Month into Datetime
df['Date']=pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# set index for date column
df.set_index('Date',inplace=True)

# visualize the data
df.plot()

'''testing for stationarity'''
#determine rolling statistics
rolmean = df.rolling(window = 5).mean()
rolstd = df.rolling(window = 5).std()

#plot rolling statistics
orig =  plt.plot(df,color ='blue',label= 'Original')
mean = plt.plot(rolmean, color = 'red',label='Rolling mean')
std = plt.plot(rolstd, color = 'black',label='Rolling std')
plt.legend(loc = 'best')
plt.title('Rolling mean & standard deviation')
plt.show(block=False)

#using augmented dicky fuller test
from statsmodels.tsa.stattools import adfuller
test_result=adfuller(df['Amount_withdrawn'])

#Ho: It is non stationary
#H1: It is stationary
#using dicky-fuller test
def adfuller_test(amount):
    result=adfuller(amount)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
        
adfuller_test(df['Amount_withdrawn'])

'''ADF Test Statistic : -6.167832856979449
p-value : 6.920539044861523e-08
#Lags Used : 3
Number of Observations Used : 241
strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary
Hence, no need of differencing and'''

#d=0

#Auto regressive model
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Amount_withdrawn'])
plt.show()


'''
Final Thoughts on Autocorrelation and Partial Autocorrelation
Identification of an AR model is often best done with the PACF.
For an AR model, the theoretical PACF “shuts off” past the order of the model. The phrase “shuts
 off” means that in theory the partial autocorrelations are equal to 0 beyond that point. Put
 another way, the number of non-zero partial autocorrelations gives the order of the AR model.
 By the “order of the model” we mean the most extreme lag of x that is used as a predictor.
Identification of an MA model is often best done with the ACF rather than the PACF.

For an MA model, the theoretical PACF does not shut off, but instead tapers toward 0 in some manner
. A clearer pattern for an MA model is in the ACF. The ACF will have non-zero autocorrelations only
 at lags involved in the model.'''

#plotting ACF and PACF plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Amount_withdrawn'],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Amount_withdrawn'],lags=40,ax=ax2)


# For non-seasonal data
#p=0, d=0, q=0
from statsmodels.tsa.arima_model import ARIMA

model=ARIMA(df['Amount_withdrawn'],order=(4,0,4))
model_fit=model.fit(disp = -1)
#model_fit.summary()

plt.plot(df)
plt.plot(model_fit.fittedvalues,color = 'red')
plt.title('RSS: %.4f'% sum((model_fit.fittedvalues-df['Amount_withdrawn'])**2))

'''df['forecast']=model_fit.predict(start=200,end=245,dynamic=True)
df[['Amount_withdrawn','forecast']].plot(figsize=(12,8))'''

import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(df['Amount_withdrawn'],order=(1,0,1),seasonal_order=(1,0,1,1))
results=model.fit(display= -1)



plt.plot(df)
plt.plot(results.fittedvalues,color = 'red')
plt.title('RSS: %.4f'% sum((results.fittedvalues-df['Amount_withdrawn'])**2))
