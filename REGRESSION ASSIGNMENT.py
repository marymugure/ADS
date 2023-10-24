#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[5]:


bs = pd.read_csv("HousingData.csv")
bs.head()


# In[ ]:





# In[8]:


target_feature = 'MEDV'
y = bs[target_feature]
X = bs.drop(target_feature, axis = 1)


# In[ ]:





# In[10]:


bs.describe()


# In[11]:


#drop ZN and CHAS
bs = bs.drop(["ZN", "CHAS"], axis = 1)
bs.head()


# In[41]:


bs.hist(figsize=(12, 10))
plt.show()


# In[42]:


plt.scatter(bs['RM'], bs['MEDV'])
plt.xlabel('Average number of rooms per dwelling (RM)')
plt.ylabel('Median value of owner-occupied homes in $1000s')
plt.show()


# In[44]:


corr = bs.corr()
corr


# In[47]:


sns.heatmap(corr, annot=True)
plt.show()


# In[17]:


#splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[46]:


#replacing missing data with mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# In[26]:


#fitting the model
model = LinearRegression()


# In[28]:


model.fit(X_train_imputed, y_train)


# In[35]:


y_pred = model.predict(X_test_imputed)


# In[36]:


mse = mean_squared_error(y_test, y_pred)
mse


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




