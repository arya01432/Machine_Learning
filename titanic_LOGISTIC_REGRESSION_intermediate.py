#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# # FILE HANDLING

# In[10]:


file_object=open("F:\\titanic.csv",'r+')
file_object.seek(0)
print(file_object.read())
file_object.close()


# In[12]: open as csv only first 10 examples


titanic_data=pd.read_csv('F:\\titanic.csv')
titanic_data.head(10)


# In[13]: total number of examples


print('no. of total passenger: '+str(len(titanic_data.index)))


# In[15]: draw a plot for survived=1,and not survived=0


sns.countplot(x='survived',data=titanic_data)


# In[17]: above ln[15] with sex of person


sns.countplot(x='survived',hue='sex',data=titanic_data)


# In[18]:  same above ln[15] with class of person


sns.countplot(x='survived',hue='pclass',data=titanic_data)


# In[19]: draw histogrm of age


titanic_data['age'].plot.hist()


# In[34]:  histogram with fare 


titanic_data['fare'].plot.hist(bins=20,figsize=(10,5))


# In[35]: whats features contain our data set print


titanic_data.info()


# In[37]:


sns.countplot(x='sibsp',data=titanic_data)


# In[38]:


sns.countplot(x='parch',data=titanic_data)


# ## DATA ANALYSING

# In[39]: which is null are true and viceversa


titanic_data.isnull()


# In[40]: total number of null for all featuers


titanic_data.isnull().sum()


# In[48]: draw heatmap 


sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap='viridis')


# In[50]: draw box plot of age vs class


sns.boxplot(x='pclass',y='age',data=titanic_data)


# In[51]: print top 5 features with values


titanic_data.head(5)


# In[52]: delete body as a featuer of our data set


titanic_data.drop('body',axis=1,inplace=True)


# In[53]:again print top 5 here body is not shown


titanic_data.head(5)


# In[54]:


titanic_data.drop('cabin',axis=1,inplace=True)
titanic_data.head(5)


# In[55]:


sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap='viridis')


# In[58]: all drop which are null 


titanic_data.dropna(inplace=True)


# In[59]: again draw heatmap


sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap='viridis')


# In[60]: sum after every null is deleted so sum will also be 0


titanic_data.isnull().sum()


# In[65]: 


sex=pd.get_dummies(titanic_data['sex'],drop_first=True)
sex.head(5)


# In[70]:


embark=pd.get_dummies(titanic_data['embarked'],drop_first=True)
embark.head(5)


# In[73]:


pclass=pd.get_dummies(titanic_data['pclass'],drop_first=True)
pclass.head(5)


# In[74]:


titanic_data=pd.concat([titanic_data,sex,embark,pclass],axis=1)
titanic_data.head()


# In[78]:


titanic_data.drop(['name','embarked','pclass','ticket'],axis=1,inplace=True)
titanic_data.head()


# In[79]:


titanic_data.drop('home.dest',axis=1,inplace=True)
titanic_data.head()


# In[80]:


titanic_data.drop('boat',axis=1,inplace=True)
titanic_data.head()


# ## TRAIN DATASETS

# In[81]:


x=titanic_data.drop('survived',axis=1)
y=titanic_data['survived']


# In[83]:


from sklearn.model_selection import train_test_split


# In[86]:


x_train,x_test,y_train,y_test = train_test_split(
        x,y,test_size=0.3,random_state=1)


# In[87]:


from sklearn.linear_model import LogisticRegression


# In[88]:


logmodel=LogisticRegression()


# In[90]:


logmodel.fit(x_train,y_train)


# In[91]:


predictions=logmodel.predict(x_test)


# In[92]:


from sklearn.metrics import classification_report


# In[93]:


classification_report(y_test,predictions)


# In[94]:


from sklearn.metrics import confusion_matrix


# In[95]:


confusion_matrix(y_test,predictions)


# In[96]:


from sklearn.metrics import accuracy_score


# In[97]:


accuracy_score(y_test,predictions)

