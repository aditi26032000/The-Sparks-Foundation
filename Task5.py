#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#read data
df1 = pd.read_csv(r'C:\Users\Alok Agrawal\Desktop\deliveries.csv')
df2 = pd.read_csv(r'C:\Users\Alok Agrawal\Desktop\matches.csv')


# In[4]:


df1.head()


# In[5]:


#checking for null values
df1.isnull().sum()


# In[6]:


df1.shape


# In[7]:


#Since the missing values in columns are hhuge in number we will drop that columns
df1.drop(columns=['player_dismissed','dismissal_kind','fielder'],inplace=True)


# In[8]:


df2.head()


# In[9]:


#check for missing values
df2.isnull().sum()


# In[10]:


df2.shape


# In[12]:


#drop umpire 3 column
df2.drop(columns=['umpire3'],inplace=True)


# In[15]:


df2.dtypes


# In[16]:


#fill missing values
df2['city'].fillna(method='ffill',inplace=True)
df2['winner'].fillna(method='ffill',inplace=True)
df2['player_of_match'].fillna(method='ffill',inplace=True)
df2['umpire1'].fillna(method='ffill',inplace=True)
df2['umpire2'].fillna(method='ffill',inplace=True)


# In[17]:


#check for null values again
df2.isnull().sum()


# In[18]:


print("Basic Overview Of Matches Dataset : \n")
print('Number Of Matches Played :',df2.shape[0])
print("Number Of Seasons Played : ",df2['season'].value_counts().nunique())
print("Top 10 Prominent Players of IPL : \n", df2['player_of_match'].value_counts()[:10])
print("Most Winning Team and Number Of Matches: \n",df2['winner'].value_counts())
print("Most Winning Team: \n",df2['winner'].value_counts().idxmax())
print("Player Of The Match & Number Of Matches : \n",df2['player_of_match'].value_counts())
print("Player Of The Match For Max . Matches : \n",df2['player_of_match'].value_counts().idxmax())


# In[19]:


print('\n')
#Some Condtional Filtering :
big_margin=df2[(df2['win_by_runs']>=100) | (df2['win_by_wickets']>=8)]
print(big_margin.winner.value_counts())


# In[20]:


print("Number Of Seasons Played IN Different Cities : \n",df2.groupby('city')['season'].nunique())
print("Number Of Winners In Different Cities \n",df2.groupby('city')['winner'].nunique())
print("Winners in Cities \n",df2.groupby('city')['winner'].value_counts())


# In[21]:



print("Match where team won by highest runs",df2.iloc[df2['win_by_runs'].idxmax()])
print('\n')
print("Match where team won by highest wickets",df2.iloc[df2['win_by_wickets'].idxmax()])


# In[22]:


print("Basic Overview of Deliveries Dataset : \n")
print(df1.info())


# In[23]:


print("Number Of Innings And Their Counts : \n",df1['inning'].value_counts())
print("Batting Team 's Max Counts :",df1['batting_team'].value_counts())
print("Number Of Super Over Matches  : \n",df1['is_super_over'].value_counts())


# In[24]:


df2.corr()['win_by_runs'].sort_values(ascending=False)


# In[25]:


df2.corr()['win_by_wickets'].sort_values(ascending=False)


# In[26]:


sns.heatmap(df2.corr(),annot=True)


# In[27]:


df2['season'].plot(kind="kde")


# In[28]:


sns.countplot(df1['is_super_over'])


# In[29]:


df1['over'].value_counts().plot(kind="bar")


# In[30]:


df1['total_runs'].value_counts().plot(kind="bar")
plt.title('Number of total runs in Different Seasons')


# In[31]:


df1['wide_runs'].value_counts().plot(kind="bar")
plt.title("Wide Runs Scored In Matches")


# In[32]:


df1.corr()['total_runs'].sort_values(ascending=False)


# In[33]:


df1['over'].value_counts()
sns.countplot(df1['over'])


# In[34]:


df1['noball_runs'].value_counts()


# In[ ]:




