#!/usr/bin/env python
# coding: utf-8

# ### 1-male,0-female; 1-poor,2-middle,3-rich;1-educated,0-not educated

# In[34]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# In[62]:


datasets=pd.read_csv("F:\\itself.csv")
datasets.head(15)


# In[36]:


print('total no.:'+str(len(datasets.index)))


# In[37]:


sns.countplot(x='Education',data=datasets)


# In[38]:


sns.countplot(x='Education',hue='Gender',data=datasets)


# In[39]:


sns.countplot(x='Education',hue='Background',data=datasets)


# In[40]:


datasets['Age'].plot.hist()


# In[41]:


datasets['Age'].plot.hist(bins=20,figsize=(20,10))


# In[42]:


datasets.info()


# In[43]:


datasets.isnull()


# In[44]:


datasets.isnull().sum()


# In[45]:


sns.heatmap(datasets.isnull(),yticklabels=False,cmap='viridis')


# In[46]:


sns.boxplot(x='Background',y='Age',data=datasets)


# In[47]:


datasets.head(5)


# In[63]:


datasets.drop(['Name','Home','School'],axis=1,inplace=True)


# In[64]:


datasets.head()


# In[65]:


x=datasets.drop('Education',axis=1)
y=datasets['Education']


# In[66]:


print(x)


# In[67]:


print(y)


# In[68]:


from sklearn.model_selection import train_test_split


# In[92]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)


# In[93]:


from sklearn.linear_model import LogisticRegression


# In[94]:


logmodel=LogisticRegression()


# In[95]:


logmodel.fit(x_train,y_train)


# In[96]:


predictions=logmodel.predict(x_test)


# In[97]:


from sklearn.metrics import classification_report


# In[98]:


classification_report(y_test,predictions)


# In[99]:


from sklearn.metrics import confusion_matrix


# In[100]:


confusion_matrix(y_test,predictions)


# In[101]:


from sklearn.metrics import accuracy_score


# In[102]:


accuracy_score(y_test,predictions)


# In[103]:


file_object=open("F:\\itself.csv",'r+')
file_object.seek(0)
print(file_object.read())
file_object.close()

