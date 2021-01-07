#!/usr/bin/env python
# coding: utf-8

# In[5]:


#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#read data 
df = pd.read_csv(r'C:\Users\Alok Agrawal\Downloads\SampleSuperstore.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[8]:


#see how variables are related to each other
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot= True)
plt.show()


# In[11]:


#check the shape of dataset
df.shape


# In[12]:


#chech how many unique category is there in each varibale
print(df.nunique())


# In[13]:


#dropping unnecessary column
#since there is only one state we can drop it
df.drop(columns=['Country'],inplace=True)


# In[19]:


#Now we will visualize profit in terms of diff variable
#1.State
plt.figure(figsize=(30,10))
plt.bar(df['State'],df['Profit'])
plt.xlabel('State')
plt.ylabel('Profit')
plt.title('Profit Vs State')
plt.show()


# In[20]:


#2.Region
plt.figure(figsize=(30,10))
plt.bar(df['Region'],df['Profit'])
plt.xlabel('Region')
plt.ylabel('Profit')
plt.title('Profit Vs Region')
plt.show()


# In[22]:


#3.Category
plt.figure(figsize=(20,10))
plt.bar(df['Category'],df['Profit'])
plt.xlabel('Category')
plt.ylabel('Profit')
plt.title('Profit Vs Category')
plt.show()


# In[23]:


#4.Sub-Category 
plt.figure(figsize=(20,10))
plt.bar(df['Sub-Category'],df['Profit'])
plt.xlabel('Sub-Category')
plt.ylabel('Profit')
plt.title('Profit Vs Sub-Category ')
plt.show()


# In[24]:


#5.Ship Mode
plt.figure(figsize=(20,10))
plt.bar(df['Ship Mode'],df['Profit'])
plt.xlabel('Ship Mode')
plt.ylabel('Profit')
plt.title('Profit Vs Ship Mode ')
plt.show()


# In[25]:


#6.Segment
plt.figure(figsize=(20,10))
plt.bar(df['Segment'],df['Profit'])
plt.xlabel('Segment')
plt.ylabel('Profit')
plt.title('Profit Vs Segment')
plt.show()


# In[32]:


#7.Sales
plt.figure(figsize = (50,20))
sns.lineplot('Sales', 'Profit', data = df, color = 'r', label= 'Sales')
plt.legend()


# In[28]:


#8.Discount
plt.figure(figsize = (10,4))
sns.lineplot('Discount', 'Profit', data = df, color = 'r', label= 'Discount')
plt.legend()


# In[29]:


#9.Quantity
plt.figure(figsize = (10,4))
sns.lineplot('Quantity', 'Profit', data = df, color = 'r', label= 'Quantity')
plt.legend()


# In[ ]:




