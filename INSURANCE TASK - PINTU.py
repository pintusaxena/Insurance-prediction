#!/usr/bin/env python
# coding: utf-8

# ## MEDICAL  INSURANCE -  DATA ANAlYSIS AND MACHINE LEARNING TECHNIQUES

# ## $\color{red}{\text {............................ 1ST IMPORTANT THING DO BEFORE START ............................ }}$

# ## $\color{green}{\text{we have to Study what the Data about and look at the picture and plan the process for solutions}}$

# # $\color{red}{\text{FEW MAJOR STEPS TO DO WITH THIS TASK}}$
# #### 1. Import Libraries
# #### 2. Get the data
# #### 3. Discover and Visualize the data to gain insights
# #### 4. Prepare the data for Machine Learning algorithms
# #### 5. Select a model and train it
# #### 6. Fine-tune your model
# 

# # $\color{Red}{\text{1st Step }}$

# `IMPORT LIBRARIES`

# In[22]:


import numpy as np # we use numerical python for understanding the data in array form


# In[23]:


import pandas as pd # we use pandas for work with data frames


# In[24]:


import matplotlib.pyplot as plt # we use matplotlib for data visualization


# In[25]:


import seaborn as sns # we use seaborn for some other task and graphical measurnment 


# In[26]:


import sklearn # we use sklearn for machine learning tasks


# # $\color{Red}{\text{ 2nd Step}}$

# `Get The Data`

# Kaggle Link - https://www.kaggle.com/mirichoi0218/insurance
# 
# -----DOWNLOADED THE DATA IN ZIP FILE---- 

# In[27]:


import zipfile


# In[28]:


fp = zipfile.ZipFile('insurance.zip')


# In[29]:


fp.filelist


# In[30]:


fp.filelist[0].filename


# NOTE - In this step we import zipfile and download the data from kaggle

# # $\color{Red}{\text{ 3rd Step}}$

# ` Data Analysis and Data Visualisation part `

# In[31]:


df = pd.read_csv('insurance.csv')  


# load the csv data and name as df

# In[32]:


df.shape


#   this column tell us the shape of data df

# In[33]:


df.info()


#  this is very informative part in this we have to understand the information given by data

# # `Above information tell us the columns details and information`
#     In this three coulns are of object type --
#     1. sex
#     2. region
#     3. smoker
#     
#     Remaing columns of Int and float types
#     
#     1.Age
#     2.BMI
#     3.Charges
#     4.Children

# 

# In[34]:


df.describe()


# ###  `Describe function is the important function in this we have see all the stastical part here. `
# ###  `In this quartile function also split and publised clearly`
# ### `25 percent , 50 percent , 75 percent is the three quartile range of it.`
# ### `mean position is always consider as a average of any data set of different features`
# ### `count function is here represent the total number of attributes of their coloumn`
#  
#  

# 

# In[35]:


df.head()


# # `Head function show the top 5 rows by default`
# 
# you can see any number also by passing parameter also - like head(20) - in this 20 top row will visible

# In[ ]:





# In[36]:


df.tail()


# # `Tail function show the bottom 5 rows by default`
# 
# you can see any number also by passing parameter also - like tail(20) - in this 20 bottom row will visible

# In[ ]:





# In[37]:


df.isnull().sum()


# ##  ` This is the very important step in this we have  seen null values ` 
# ## ` null values is always a worry point for the dataset`
# ### `Fortunately we dont have any null values in this data`

# In[38]:


df['sex'].value_counts()


# In[39]:


plt.figure(figsize=(14,8) ,dpi = 80 , facecolor='black')
df['sex'].value_counts().plot(kind='bar' ,color = ['yellow','cyan'] , width = 0.3 )
plt.xlabel('Gender' , fontsize = 25 , color = 'cyan')
plt.ylabel(' Total Male and Total Female ' , fontsize = 20 , color = 'cyan' )
count = 0
for i in df['sex'].value_counts():
    plt.text(count-0.05 , i+11, i , fontsize = 12 , color = 'blue')
    count+=1
plt.xticks(fontsize = 25 , color = 'white')
plt.yticks(fontsize = 20 , color = 'white')
plt.show()


# ## In above graph , using pandas function with matplotlib visualitaion 
# ## This is the bar graph between Male and Female , total number of Male and Female

# In[ ]:





# In[40]:


plt.figure(figsize=(5,4) , dpi = 100 , facecolor='cyan')
df['age'].value_counts().plot(kind='box')
plt.xlabel('AGE' , fontsize = 15 , color = 'black')
plt.yticks(color = 'black')
plt.show()


# `This is the box plot in this we seen some outlier duw to which this data is skewd up side`

# In[ ]:





# # `Some Histogram Data plot`

# # ....................... `HIST AND BAR GRAPH BTWN AGE AND BMI `.................

# In[41]:


fig , (ax1 , ax2) = plt.subplots(1,2,dpi=150)
ax1.hist(df['age'] , bins = 10 , color = 'red', label = 'age' , ec = 'gold')
ax1.legend()
ax2.hist(df['bmi'] , bins = 15 , color = 'green' , label ='bmi' ,ec = 'gold')
ax2.legend()
plt.show()


# # ....................... `HIST GRAPH BTWN AGE AND BMI `.................

# In[42]:


fig , (ax1 , ax2) = plt.subplots(1,2,dpi=150)
ax1.hist(df['age'] , bins = 10 , color = 'red', label = 'age' , ec = 'gold')
ax1.legend()
ax2.scatter(df['age'] ,df['bmi'] , color = 'green' , label ='bmi' )
ax2.legend()
plt.show()


# In[43]:


fig , (ax1 , ax2) = plt.subplots(1,2,dpi=100)
ax1.hist(df['age'] , bins = 10 , color = 'blue', label = 'age' , ec = 'gold')
ax1.legend()
ax2.scatter(df['age'] ,df['smoker'] ,  color = 'cyan' , label ='bmi' )
ax2.legend()
plt.show()


# # $\color{Red}{\text{ 4rth Step}}$

# ` EDA AND PREPARATION OF DATA ` 

# # $ \color{red}{\text{.......................we have to find correlation now....................... }} $

# In[44]:


df.corr()


# # `In this we have to seen the relaion of diiferent attributs which each other`
# ## ` But this data is only for numerical values`

# In[ ]:





# # `CORRELATION GRAPH `

# In[45]:


plt.figure(figsize=(8,6) , dpi = 100)
sns.heatmap(df.corr(),annot = True,fmt=".2f" , cmap ='PiYG_r');


#  `In this we only see numerical data types value and we have to decide our important columns and features`

# 

# 

#  ${\text{ WE   CONSIDER}}$  $\color{red}{\text{CHARGES}}$${\text{  AS  LABEL  IN  DATA SET }} $

# In[46]:


plt.figure(dpi=130)
sns.scatterplot(df['bmi'] , df['charges'] , hue= df['sex'])


# ## `This graph shown us the scatter plot between bmi and charges and you see clearly that both male and female have a insurance range in 10k to 20 k`

# 

# 

# # `BOXPLOT`

# In[29]:


plt.figure(dpi=120)
sns.boxplot(x=df['charges'])


# ## ` IN THIS WE SEE THAT SOME DATA HAVE HIGH VALUES BUT MOST OF THE DATA CHARGES IS IN RANGE OF 10K TO 20 K ONLY`

# 

# 

# In[30]:


plt.figure(dpi = 100)
plt.scatter(df['age'] , df['charges'])
print(np.corrcoef(df['age'],df['charges']))
plt.show()


# # `IN THIS AGE AND CHARGES HAVE 29 PERCENT IMPACT ON EACH OTHER ...... ITS A POSITIVE CORELATION`

# In[ ]:





# # $\color{red}{\text{Some Histogram Plot with Corelations}}$

# In[31]:


plt.figure(dpi=100)
df.hist(bins=30, ec='k', color = 'cyan' , figsize=(15, 10))
plt.show()


# In[32]:


plt.figure(figsize=(6,5) , dpi = 100)
sns.barplot(x = 'sex' , y ='charges' , data=df)


# In[33]:


plt.figure(figsize=(16,5) , dpi = 150 , facecolor='salmon')
plt.xticks(color = 'white' , fontsize = 15 , rotation = 90)
plt.yticks(color = 'white' , fontsize = 15)

sns.barplot(x = 'age' , y ='charges' , data=df  )
plt.show()


# 

# $ \color{red}{\text{************** NOW WE HAVE PLAY WITH CATAGORICAL COLUMNS*****************}}$

# In[34]:


df['region'] = pd.Categorical(df['region'])


# In[35]:


df['smoker'] = pd.Categorical(df['smoker'])


# In[36]:


df['sex'] = pd.Categorical(df['sex'])


# In[37]:


df_num = df.drop(['smoker','sex','region' ],axis=1)


# In[38]:


df_num.head()


# In[39]:


df['region'].unique()


# In[40]:


df['smoker'].unique()


# In[41]:


df['sex'].unique()


# In[42]:


from sklearn.preprocessing import OrdinalEncoder


# In[43]:


df_cat = df[['smoker' , 'sex' , 'region']]
df_cat.head()


# In[44]:


ordinal_encoder = OrdinalEncoder()


# In[45]:


ordinal_encoder.fit(df_cat)


# In[46]:


ordinal_encoder.categories_


# In[47]:


df_cat_encoding = ordinal_encoder.transform(df_cat)


# In[48]:


df_cat_encoding


# In[49]:


df_cat.head()


# from ordinal encoder we cant get proper result so i used label encoding

# In[50]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[51]:


df["sex"] = label_encoder.fit_transform(df["sex"])
df["smoker"] = label_encoder.fit_transform(df["smoker"])
df["region"] = label_encoder.fit_transform(df["region"])


# In[52]:


df.info()


# # Now data change so we will see final correlation of different data

# In[53]:


df.corr()


# In[54]:


df[['sex']]


# In[55]:


plt.figure(dpi=150)
sns.heatmap(df.corr() , cmap = 'Set1_r' , alpha = 0.9)


# In[56]:


plt.figure(dpi=150)
df.hist(bins=30, ec='k', color = 'salmon' , figsize=(15, 10))
plt.show()


# ## $ {\text{IN THIS WE FIND THAT NOW}}$ $ \color{red}{\text {SMOKER IS THE HIGHERST CORRELATION WITH OUR LABEL }}$

# In[57]:



sns.swarmplot(x=df['smoker'],
              y=df['charges'] )
plt.xticks(fontsize = 15)

#HERE
# 0 mean NO
# 1 MEANS YES


# `In this we see that the person those do smoking have more chance to get high value insurance carges`

# In[58]:


plt.figure(dpi = 150)
sns.scatterplot(x=df['bmi'], y=df['charges'], hue=df['smoker'])
plt.show()


# ` in this we see that the person those do smoking have more chance to get high value insurance carges with same BMI index `

# In[59]:


plt.figure(dpi = 150)
sns.scatterplot(x=df['age'], y=df['charges'], hue=df['smoker'])
plt.show()


# # $\color{Red}{\text{ 5th Step}}$

# `Select the Model and Train it `

# In[60]:


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


# ### WE HAVE TO IMPORT LIBRARIES FOR MACHINE LEARNING MODEL FROM SKLERN 
# 
# ### TRAIN AND TEST DATA SPLIT - WE USE train_test_split
# 
# ### FOR REGRESSION - WE USE LINERA REGRESSION MODEL
# 
# ### MEAN SSQUARE ERROR FOR FINDIND THE ERROR

# In[ ]:





# # Split data into x and y

# In[61]:


#select features and lables
x = df.drop(['charges'], axis = 1)
y = df['charges']


# In[62]:


x.head()


# In[63]:


y.head()


# In[ ]:





# # For Splitting the Training and Testing Data

# In[64]:


#split train and Testing
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)


# In[65]:


x_train # data of train set in x 


# In[66]:


x_test # data of test set in x 


# In[67]:


y_train # data of train set in y 


# In[68]:


y_test # data of test set in y


# In[69]:


x_train.shape


# In[70]:


x_test.shape


# In[71]:


corr_matrix = df.corr()
corr_matrix['charges'].sort_values(ascending=False)


# In[72]:


df[['smoker']].hist(bins = 40 , ec = 'gold')


# In[73]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[48]:


df['age_cat'] = pd.cut( df['age'],
bins = [ 0, 18 , 28 , 38 , 48, 58 , 68, 78, 88, np.inf],
labels = [18, 2, 3, 4, 5,6,7,8,9])


# In[75]:


#Linear regression algorithm
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[76]:


# we will remove the column "region" because it is not necesaary
df.drop(["region"],axis=1,inplace=True)


# In[77]:


# dividing the dataset into attributes and label
x1 = df.iloc[:,0:-1].values
y1 = df.iloc[:, -1].values


# In[78]:


print(x)


# In[79]:


print(y)


# In[80]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
x1 = sc_X.fit_transform(x)


# In[81]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size=0.2,random_state=42)


# In[82]:


Lin_reg = LinearRegression()


# In[83]:


# training the data
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)


# In[84]:


#predicting
y_pred = lr.predict(x_test)


# In[85]:


# we will calculate Mean Absolute error
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error:',mae)


# In[ ]:





# ## CONCLUSION --- This is the Dataset Named Insurance Prediction in this we have do Analysis 
# 
# ## ERROR - MEAN ABSOLUTE ERROR ----- 0.27
# 
# ## PERCENTAGE = 27%
# 
# ## SO ACURRACY = 100 - ERROR 
# 
# ## 100 - 27 = 63 % = ACCURACY

# In[ ]:




