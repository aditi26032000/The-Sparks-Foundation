#!/usr/bin/env python
# coding: utf-8

# In[27]:


#import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 


# In[28]:


#read dataset
df = pd.read_csv(r"https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
df.head()


# In[29]:


#visualize the data
plt.scatter(df['Hours'],df['Scores'],color = 'hotpink')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Scores Vs Hours')
plt.show()


# In[30]:


#split the data into train and test
x_train,x_test,y_train,y_test = train_test_split(df['Hours'],df['Scores'],test_size=0.2,random_state=0)


# In[31]:


x_train = x_train.values.reshape(20,1)
y_train = y_train.values.reshape(20,1)
x_test = x_test.values.reshape(5,1)
y_test = y_test.values.reshape(5,1)


# In[32]:


#build the model
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[33]:


#calculate error
print(mean_squared_error(y_test,y_pred))


# In[34]:


#Visulization of outcome
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Scores Vs Hours')
plt.show()


# In[35]:


#Predicting score for given time
target = [[9.25]]
pred = model.predict(target)
print("No of Hours = {}".format(target))
print("Predicted Score = {}".format(pred[0]))


# In[ ]:




