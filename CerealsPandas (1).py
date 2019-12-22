#!/usr/bin/env python
# coding: utf-8

# In[101]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[102]:


df = pd.read_csv("C:\\pandas\\cereals.csv")


# In[103]:


df


# In[104]:


print(df.dtypes)


# In[105]:


df.head()


# In[106]:


df.iloc[1:2]


# In[107]:


df.iloc[[1,4], [4,2]]


# In[108]:


df.loc[5:20,("mfr","fat")]


# In[109]:


df.drop("cups",axis=1)


# In[110]:


df.describe()


# In[111]:


df.describe(include=['object'])


# In[112]:


print(df.describe(include=['object']))


# In[113]:


df.drop("mfr",axis=1)


# In[114]:


#adding a column
for carbo, row in df.iterrows():
    df.loc[carbo,"new column"] =  row["fiber"]*row["rating"]


# In[115]:


df


# In[116]:


df["name"].str.split("_",n=1,expand=True)


# In[117]:


df


# In[118]:


df.head(10)


# In[119]:


df2=df.head(10)


# In[120]:


import matplotlib.pyplot as plt
plt.bar(df2['name'], df2['mfr'], align='center',color='g')
plt.xticks(rotation=90)
plt.title("cereals rough")
plt.xlabel("mfr")
plt.ylabel("ratings")
plt.show()


# # Performing Linear Regression
# 

# In[121]:


newdf = df.loc[:,['calories','protein','fat','sodium','fiber','carbo','sugars','potass','vitamins','rating']]
print(newdf.head(5))


# In[122]:


# X = feature values, all the columns except the last column
X = newdf.iloc[:, [0,1,4,6]]
# y = target values, last column of the data frame
y = newdf.iloc[:, -1]
X.head()


# In[123]:


#splitting the data into train and test in 80/20
split = int(0.8*len(newdf))
X_train, X_test, Y_train, Y_test = X[:split], X[split:], y[:split], y[split:]

import numpy as np
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, Y_train)


# In[124]:


print(reg)


# In[125]:


reg.score(X,y)


# In[126]:


newpred = reg.predict(X_test)
#type(newpred)
xyz = pd.DataFrame(newpred)


# In[127]:


xyz.head()


# In[128]:


#To conatct the new values

df_new = pd.concat([X_test, Y_test], axis=1)

df_new


# In[129]:


# Original Rating, Predict Rating
df_new['pred_rating'] = newpred
df_new
#df_new2 = pd.concat([X_test, xyz], axis=1)
#df_new2


# In[130]:


print('Coefficients: \n', reg.coef_)


# In[131]:


print('Coefficients: \n', reg.score(X,y))


# In[132]:


data = newdf
data


# In[133]:


X = data.iloc[:, 6].values.reshape(-1, 1)  # values converts it into a numpy array
Y = data.iloc[:, 9].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
plt.scatter(X, Y, color='green')
plt.plot(X, Y_pred, color='red')
plt.show()


# In[ ]:




