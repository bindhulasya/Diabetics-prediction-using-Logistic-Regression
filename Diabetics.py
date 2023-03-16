#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("diabetes2.csv")


# In[3]:


# Data manipulation libraries
import numpy as np

###scikit Learn Modules needed for Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder,StandardScaler
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder,StandardScaler 
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#for plotting
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)

import warnings
warnings.filterwarnings('ignore')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.isnull().sum()


# In[7]:


df.describe(include= "all")


# In[8]:


df.corr()


# In[24]:


#Plotting Graph for Data Visuvalization

sns.heatmap(df.corr(),annot=True)
plt.show()


# In[10]:


df.hist()
plt.show()


# In[11]:


#PreProcessing data

df.columns


# In[12]:


# For Numerical Value Pre-processing we use Standard Scalar

scaler=StandardScaler()

df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]=scaler.fit_transform(df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']])


# In[13]:


df.head()


# In[14]:


df_new=df.copy()


# In[19]:


# Train & Test split
a_train, a_test, b_train, b_test = train_test_split( df_new[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']],
         df_new['Outcome'],test_size=0.20,random_state=23)

print('Shape of Training Xs:{}'.format(a_train.shape))
print('Shape of Test Xs:{}'.format(a_test.shape))
print('Shape of Training y:{}'.format(b_train.shape))
print('Shape of Test y:{}'.format(b_test.shape))


# In[20]:


# Building Linear Regression Model

model = LogisticRegression()
model.fit(a_train, b_train)
b_predicted = model.predict(a_test)


# In[21]:


score=model.score(a_test,b_test);     #testing the linear regression model


# In[22]:


# Model diagnostic
print(score)

