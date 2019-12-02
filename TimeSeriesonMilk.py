#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from pandas import datetime
from datetime import timedelta


# In[4]:


df = pd.read_csv("C:\\pandas\\milk.csv")


# In[5]:


df.head(5)


# In[6]:


df.tail()


# In[7]:


#To change the date format to YY/MM/DD
df['month'] = pd.to_datetime(df['month'])
df.head()


# In[8]:


df.describe()


# In[9]:


df.describe().transpose()


# In[10]:


#Used to plat the grapth taking month as index

df.set_index('month',inplace=True)
df.plot()


# In[11]:


#To calculate the mean and STD for the time series
#We first use the Rolling test and later ADULF

timeseries= df['milk_prod_per_cow_kg']
timeseries.rolling(12).mean().plot(label='12 Month Rolling Mean')
timeseries.rolling(12).std().plot(label='12 Month Rolling Std')
timeseries.plot()

plt.legend()


# In[12]:


#To check the mean for 12 months
timeseries.rolling(12).mean().plot(label='12 Month Rolling Mean')
timeseries.plot()
plt.legend()


# In[13]:


#To check the STD for 12 months
timeseries.rolling(12).std().plot(label='12 Month Rolling std')
timeseries.plot()
plt.legend()


# # To check componenets or decompositions
# 
# 

# In[14]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[15]:


# Use decomposition for seasoal decompose.
#fig is used to plot figure
decomposition = seasonal_decompose(df['milk_prod_per_cow_kg'], freq=12)

#fig = plt.figure()

fig = decomposition.plot()

fig.set_size_inches(15, 8)


# # Augmented Dickey-Fuller unit root test

# In[16]:


# A small p-value (typically < 0.05) indicates strong evidence against the null hypothesis, so you reject the null hypothesis.
#Alarge p-value (> 0.05) indicates weak evidence against the null hypothesis, so you fail to reject the null hypothesis


# In[17]:


from statsmodels.tsa.stattools import adfuller
result = adfuller(df['milk_prod_per_cow_kg'])
print( "Augmented Dickey-Fuller Test:")
labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
for value,label in zip(result,labels):
    print(label+' : '+str(value) )
      
if result[1] <= 0.05:
    print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
else:
    print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# In[ ]:




