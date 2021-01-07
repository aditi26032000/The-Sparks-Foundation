#!/usr/bin/env python
# coding: utf-8

# In[10]:


#import libraries
import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('vader_lexicon')
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
import xgboost 


# In[9]:


get_ipython().system('pip install textblob')


# In[11]:


get_ipython().system('pip install pmdarima')


# #Numerical Analysis

# In[12]:


df_prices = pd.read_csv(r'C:\Users\Alok Agrawal\Downloads\CSV.csv')
print(df_prices.head())
print(df_prices.size)


# In[13]:


#Converting Date column to datetime datatype
df_prices['Date'] = pd.to_datetime(df_prices['Date'])
df_prices.info()


# In[14]:


df_prices.dropna(inplace = True)


# In[15]:


plt.figure(figsize=(10, 6))
df_prices['Close'].plot()
plt.ylabel('Close')


# In[16]:


#Plotting moving average
close = df_prices['Close']
ma = close.rolling(window = 50).mean()
std = close.rolling(window = 50).std()

plt.figure(figsize=(10, 6))
df_prices['Close'].plot(color = 'b', label = 'Close')
ma.plot(color = 'r', label = 'Rolling Mean')
std.plot(label = 'Rolling Standard Deviation')
plt.legend()


# In[17]:


#Plotting returns
returns = close / close.shift(1) - 1

plt.figure(figsize = (10,6))
returns.plot(label='Return', color = 'g')
plt.title("Returns")


# In[28]:


df_prices.shape


# In[30]:


train = df_prices[:203]
test = df_prices[203:]


# In[31]:


#Stationarity test
def test_stationarity(timeseries):

 #Determing rolling statistics
 rolmean = timeseries.rolling(20).mean()
 rolstd = timeseries.rolling(20).std()

 #Plot rolling statistics:
 plt.figure(figsize = (10,8))
 plt.plot(timeseries, color = 'y', label = 'original')
 plt.plot(rolmean, color = 'r', label = 'rolling mean')
 plt.plot(rolstd, color = 'b', label = 'rolling std')
 plt.xlabel('Date')
 plt.legend()
 plt.title('Rolling Mean and Standard Deviation',  fontsize = 20)
 plt.show(block = False)
 
 print('Results of dickey fuller test')
 result = adfuller(timeseries, autolag = 'AIC')
 labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
 for value,label in zip(result, labels):
   print(label+' : '+str(value) )
 if result[1] <= 0.05:
   print("Strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
 else:
   print("Weak evidence against null hypothesis, time series is non-stationary ")
test_stationarity(train['Close'])


# In[32]:


train_log = np.log(train['Close']) 
test_log = np.log(test['Close'])

mav = train_log.rolling(24).mean() 
plt.figure(figsize = (10,6))
plt.plot(train_log) 
plt.plot(mav, color = 'red')


# In[33]:


train_log.dropna(inplace = True)
test_log.dropna(inplace = True)

test_stationarity(train_log)


# In[34]:


train_log_diff = train_log - mav
train_log_diff.dropna(inplace = True)

test_stationarity(train_log_diff)


# In[35]:


test.head()


# In[37]:


#Using auto arima to make predictions using log data
from pmdarima import auto_arima
model = auto_arima(train_log, trace = True, error_action = 'ignore', suppress_warnings = True)
model.fit(train_log)
predictions = model.predict(n_periods = len(test))
predictions = pd.DataFrame(predictions,index = test_log.index,columns=['Prediction'])


# In[38]:


plt.plot(train_log, label='Train')
plt.plot(test_log, label='Test')
plt.plot(predictions, label='Prediction')
plt.title('BSESN Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')


# In[39]:


#Calculating error
rms = np.sqrt(mean_squared_error(test_log,predictions))
print("RMSE : ", rms)


# #Sentiment Analysis on News Headlines

# In[40]:


#read data
df_news = pd.read_csv(r'C:\Users\Alok Agrawal\Downloads\india-news-headlines.csv')
df_news.head()


# In[41]:


df_news.describe()


# In[45]:



df_news.drop(columns=['headline_category'], inplace=True)
df_news.info()


# In[47]:


#Converting data type of Date column 
df_news['Date'] = pd.to_datetime(df_news['publish_date'],format= '%Y%m%d')
df_news


# In[48]:


#Grouping the headlines for each day
df_news['News'] = df_news.groupby(['Date']).transform(lambda x : ' '.join(x)) 
df_news = df_news.drop_duplicates() 
df_news.reset_index(inplace = True, drop = True)
df_news


# In[50]:


ps = PorterStemmer()


# In[ ]:


#Cleaning headlines
c = []
for i in range(0,len(df_news['News'])):
    news = re.sub('[^a-zA-Z]',' ',df_news['News'][i])
    news = news.lower()
    news = news.split()
    news = [ps.stem(word) for word in news if not word in set(stopwords.words('english'))]
    news=' '.join(news)
    c.append(news)


# In[ ]:


df_news['News'] = pd.Series(c)
df_news


# In[ ]:


#Functions to get the subjectivity and polarity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return  TextBlob(text).sentiment.polarity


# In[ ]:


#Adding subjectivity and polarity columns
df_news['Subjectivity'] = df_news['News'].apply(getSubjectivity)
df_news['Polarity'] = df_news['News'].apply(getPolarity)
df_news


# In[ ]:


plt.figure(figsize = (10,6))
df_news['Polarity'].hist(color = 'purple')


# In[ ]:


plt.figure(figsize = (10,6))
df_news['Subjectivity'].hist(color = 'blue')


# In[ ]:


sia = SentimentIntensityAnalyzer()

df_news['Compound'] = [sia.polarity_scores(v)['compound'] for v in df_news['News']]
df_news['Negative'] = [sia.polarity_scores(v)['neg'] for v in df_news['News']]
df_news['Neutral'] = [sia.polarity_scores(v)['neu'] for v in df_news['News']]
df_news['Positive'] = [sia.polarity_scores(v)['pos'] for v in df_news['News']]
df_news


# In[ ]:


df_merge = pd.merge(df_prices, df_news, how='inner', on='Date')
df_merge


# In[ ]:


df = df_merge[['Close','Subjectivity', 'Polarity', 'Compound', 'Negative', 'Neutral' ,'Positive']]
df.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
new_df = pd.DataFrame(sc.fit_transform(df))
new_df.columns = df.columns
new_df.index = df.index
new_df.head()


# In[ ]:


X = new_df.drop('Close', axis=1)
y =new_df['Close']


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
x_train.shape


# In[ ]:


#Build model
xgb = xgboost.XGBRegressor()
xgb.fit(x_train, y_train)


# In[ ]:


predictions = xgb.predict(x_test)
print(mean_squared_error(predictions,y_test))

