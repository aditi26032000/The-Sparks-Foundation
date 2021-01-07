#!/usr/bin/env python
# coding: utf-8

# In[20]:


#import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[21]:


# read data
df = pd.read_csv(r'C:\Users\Alok Agrawal\Downloads\Iris.csv')
df.head()


# In[22]:


#defining feature and labels
x = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df['Species']


# In[23]:


#Encoding the target variable
encoder = LabelEncoder()
y = encoder.fit_transform(y)


# In[24]:


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)


# In[25]:


#building the decission tree classifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)


# In[29]:


#Visualizing the decission tree
plt.figure(figsize=(25,20))
tree.plot_tree(model, feature_names=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'],class_names=['Iris-setosa','Iris-versicolour','Iris-virginica'],filled=True)
plt.show()


# In[30]:


#predicting the outcome
y_pred = model.predict(x_test)


# In[31]:


#Calculating the accuracy
print(accuracy_score(y_test, y_pred))


# In[ ]:




