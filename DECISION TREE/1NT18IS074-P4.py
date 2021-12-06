#!/usr/bin/env python
# coding: utf-8

# # DECISION TREE PROGRAM 4 1NT18IS074

# In[2]:


import pandas as pd
import warnings
from sklearn import tree
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
df=pd.read_csv(r'/home/admin1/Desktop/zoo.csv')
print (df)


# In[4]:


x= df.iloc[:, 1:17]
x.shape
print(df.iloc[:, 1:17])


# In[5]:


y=df.iloc[:,17]
y.shape


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.8,test_size=.2)
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier().fit(x_train,y_train)
pred=clf.predict(x_test)
accuracy=clf.score(x_test,y_test)
print('Accuracy ', accuracy)


# In[7]:


from sklearn import metrics 
print(metrics.confusion_matrix(y_test,pred))


# In[8]:


print(metrics.classification_report(y_test,pred))


# In[9]:


plt.figure(figsize=(15,15))
tree.plot_tree(clf)
plt.show()


# In[10]:


df.head()


# In[ ]:




