#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# In[2]:


df = pd.read_csv("C:\\pandas\\titanic.csv")


# In[3]:


df.head(10)


# In[4]:


df.describe()


# In[5]:


print("Number of passangers:" +str(len(df.index)))


# In[6]:


#count plot
sns.countplot(x="Survived", data=df)


# In[7]:


sns.countplot(x="Age", data=df)
sns.countplot(x="Pclass", data=df)


# In[8]:


sns.countplot(x="Survived", hue="Sex" ,data=df)


# In[9]:


sns.countplot(x="Survived", hue="Pclass", data=df)


# In[10]:


df["Age"].plot.hist()


# In[11]:


df["Fare"].plot.hist(bins=20, figsize=(10,5))


# In[12]:


df.info()


# In[13]:


sns.countplot(x="SibSp", data=df)


# In[14]:


df.isnull()


# In[15]:


df.isnull().sum()


# In[16]:


sns.boxplot(x="Pclass", y="Age", data=df)


# In[17]:


#Imputation


# In[18]:


df.head(10)


# In[19]:


df.drop("Cabin", axis=1)


# In[20]:


df.dropna(inplace=True)


# In[21]:


df.isnull().sum()


# In[22]:


df.head(5)


# In[25]:


#Craeting dummies

newsex=pd.get_dummies(df["Sex"], drop_first=True)
sex.head(5)


# In[24]:


newembark= pd.get_dummies(df["Embarked"], drop_first=True)
embark.head(5)


# In[62]:


pcl= pd.get_dummies(df["Pclass"], drop_first=True)
pcl.head(5)


# In[63]:


df.head(5)


# In[64]:


df=pd.concat([df,newsex,pcl,newembark],axis=1)


# In[65]:


df.head(10)


# In[66]:


df.head(5)


# In[67]:


df.drop(['Name','Sex','PassengerId','Ticket','Pclass'],axis=1, inplace=True)


# In[68]:


df.head(5)


# splitting Data

# In[69]:


X=df.drop("Survived", axis=1)
y=df["Survived"]


# In[70]:


from sklearn.model_selection import LogisticRegression


# In[71]:


train_test_split


# In[72]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)


# In[73]:


from sklearn.linear_model import LogisticRegression


# In[74]:


logmodel=LogisticRegression()


# In[75]:


logmodel.fit(X_train,y_train)

