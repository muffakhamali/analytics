#!/usr/bin/env python
# coding: utf-8

# In[186]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns


# In[187]:


df = pd.read_csv("C:\\pandas\\blackfriday_train.csv")


# In[188]:


df.head(10)


# In[189]:


print("Number of users:" +str(len(df.index)))


# In[190]:


#To remove the unncessary values
df['Age']=df["Age"].str.extract('(\d\d)',expand=True)


# In[191]:


df.head(5)


# In[192]:


df.info()


# In[193]:


df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].map(lambda x: x.lstrip('+').rstrip(' '))


# In[194]:


df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].str.replace(r'\D', '')


# In[195]:


df.head(5)


# In[196]:


df.info()


# In[197]:


#To remove the unncessary values
df['Age']=df["Age"].str.extract('(\d\d)',expand=True)


# In[198]:


plt.style.use("fivethirtyeight")
plt.figure(figsize=(12,7))
sns.distplot(df.Purchase, bins = 25)
plt.xlabel("Amount spent in Purchase")
plt.ylabel("Number of Buyers")
plt.title("Purchase amount Distribution")


# In[199]:


sns.countplot(x="Occupation", data=df)


# In[200]:


sns.countplot(x="Marital_Status", data=df)


# In[201]:


sns.countplot(x="Age", data=df)


# In[202]:


sns.countplot(x="Gender", data=df)


# In[203]:


sns.countplot(df.Age)


# In[204]:


sns.countplot(df.City_Category)


# In[205]:


sns.countplot(df.Stay_In_Current_City_Years)


# In[206]:


df.info()


# In[207]:


sns.countplot(x="Product_Category_1", data=df)


# In[208]:


sns.countplot(x="Product_Category_2", data=df)


# In[209]:


sns.countplot(x="Product_Category_3", data=df)


# In[210]:


#To find the correlation:

df.corr(method ='pearson') 


# In[211]:


#highest correlation we have is for "occupation"
#Has highest negative co relation Product_Category_1


# In[212]:


df.head(10)


# In[213]:


df.isnull().sum()


# In[214]:


print(df.shape)


# In[215]:


sns.heatmap(df.isnull(), cbar=False)


# In[216]:


import missingno as msno


# In[217]:



msno.matrix(df)


# In[218]:


df['Product_Category_3'].fillna(0, inplace=True)


# In[219]:


df['Product_Category_2'] = df['Product_Category_2'].replace(np.nan, 0)


# In[220]:


df.head(5)


# In[221]:


df.info()


# In[222]:


df["Product_Category_2"]= df["Product_Category_2"].astype(int) 


# In[223]:


df["Product_Category_3"]= df["Product_Category_3"].astype(int) 


# In[224]:


df.head(5)


# In[225]:


df.info()


# In[226]:


df["Age"]= df["Age"].astype(int) 


# In[227]:


df.info()


# In[228]:


df.head(5)


# In[229]:


sns.countplot(x="Product_Category_2", hue="Age", data=df)


# In[230]:


sns.countplot(x="Product_Category_2", hue="Gender", data=df)


# In[231]:


plt.hist(df['Product_Category_2'], bins=10, color='green')


# In[232]:


df.head(5)


# In[233]:


#Imputation
sex=pd.get_dummies(df["Gender"], drop_first=True)
sex.head(5)


# In[234]:


city=pd.get_dummies(df["City_Category"], drop_first=True)
city.head(5)


# In[235]:


df=pd.concat([df,sex,city,],axis=1)


# In[236]:


df.head(10)


# In[237]:


df.drop(['Gender','City_Category'],axis=1, inplace=True)


# In[238]:


df.head(5)


# In[239]:


df.drop(["Marital_Status"], axis=1)


# In[240]:


df.head(5)


# In[241]:


df["Stay_In_Current_City_Years"]= df["Stay_In_Current_City_Years"].astype(int) 


# In[242]:


df.info()


# In[243]:


#Training the Data


# In[248]:


df.drop("Product_ID",axis=1)


# In[250]:


df.drop("User_ID", axis=1)


# In[262]:


X=df.drop("Purchase", axis=1)
y=df["Purchase"]


# In[263]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)


# In[264]:


#Creating model


# In[265]:


from sklearn.linear_model import LinearRegression


# In[266]:


df.head(5)


# In[267]:


lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.fit(X_train, y_train))


# In[270]:


predictions = lm.predict(X_test)
print("Prediction for new costumers:", predictions)


# In[273]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))


# In[316]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_test, predictions))
print(rmse)


# #Decison Tree

# In[ ]:




