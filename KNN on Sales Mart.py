#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[78]:


df = pd.read_csv("C:\\pandas\\train.bigmart.csv")


# In[79]:


df.head(5)


# In[80]:


# we need to first find the missing values
df.isnull().sum()


# In[81]:


#Now we have found that there are missing values in item weight and item outlet.

mean =df['Item_Weight'].mean()
df['Item_Weight'].fillna(mean, inplace=True)

#lets take mode for outlet size
mode = df['Outlet_Size'].mode()
df['Outlet_Size'].fillna(mode[0], inplace=True)


# In[82]:


df.isnull().sum()


# In[83]:


#To drop columns which are not requred.
df.info()


# In[84]:


#Dropping item indentifier and outlet identifier
df.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1, inplace=True)
df = pd.get_dummies(df)


# In[85]:


#Now that we have completed the EDA.
#we can proceed with model buidling

from sklearn.model_selection import train_test_split
train , test = train_test_split(df, test_size=0.3)

x_train= train.drop('Item_Outlet_Sales', axis=1)
y_train=train['Item_Outlet_Sales']

x_test= test.drop('Item_Outlet_Sales', axis=1)
y_test=test['Item_Outlet_Sales']


# In[86]:


#Scaling plays a vital role in KNN, to either standardize,normalise,minmax,roubust the data.

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)


# In[87]:


#To find the error rate in the k value.
#import required packages

from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[88]:


rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(x_train, y_train)  #fit the model
    pred=model.predict(x_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


# In[89]:


#to plot the RMSE values in graph (Elbow curve)

curve = pd.DataFrame(rmse_val)
curve.plot()


# In[66]:


#Looking at the curve, we cans say the at K=1 the values have a high RMSE and
#at k=20 the value had a very low rmse. Hence, We can say that at k=7 the k value gives the best result.


# In[98]:


test = pd.read_csv("C:\\pandas\\test.bigmart.csv")
test.head(5)


# In[104]:


submission = pd.read_csv("C:\\pandas\\submission.csv")
submission


# In[105]:


submission['Item_Identifier'] = test['Item_Identifier']
submission['Outlet_Identifier'] = test['Outlet_Identifier']


# In[110]:


#preprocessing test dataset
test.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1, inplace=True)
test['Item_Weight'].fillna(mean, inplace =True)
test = pd.get_dummies(test)
test_scaled = scaler.fit_transform(test)
test = pd.DataFrame(test_scaled)


# In[112]:


#predicting on the test set and creating submission file
predict = model.predict(test)
submission['Item_Outlet_Sales'] = predict
submission.to_csv('submit_file.csv',index=False)


# In[113]:


from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}

knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)
model.fit(x_train,y_train)
model.best_params_


# In[ ]:




