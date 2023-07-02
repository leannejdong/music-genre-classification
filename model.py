#!/usr/bin/env python
# coding: utf-8

# ## Analyse the data using pandas

# In[127]:


import pandas as pd
import numpy as np
data = pd.read_csv('data.csv')
data.head()


# In[128]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
sns.catplot(y= 'label', kind= 'count', data= data, palette = 'pastel')


# In[129]:


data.shape


# In[130]:


data = data.drop(['filename'], axis=1) # drop name of the column which is unnecessary


# ## Encoding the labels

# In[131]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
labels = data.iloc[:,-1]
encoder = LabelEncoder()
y = encoder.fit_transform(labels)


# ## Scaling the feature columns

# In[132]:


scaler=StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))


# ## split the dataset 

# In[133]:


train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
train_x.shape, test_x.shape, train_y.shape, test_y.shape
train_x.shape[1]


# In[134]:


#reshaping into 2d array
train_x=np.reshape(train_x,(train_x.shape[0],13,2,1))
test_x=np.reshape(test_x,(test_x.shape[0],13,2,1))
train_x.shape, test_x.shape, train_y.shape, test_y.shape


# In[135]:


from keras.utils.np_utils import to_categorical
train_y=to_categorical(train_y, num_classes=10)
test_y=to_categorical(test_y, num_classes=10)
train_y.shape, test_y.shape
train_y


# ## Building our neural network

# In[136]:


from keras import Sequential
# import BatchNormalization
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
model=Sequential()


# In[137]:


#adding layers and forming the model
model.add(Conv2D(64,kernel_size=5,strides=1,padding="same",activation="relu",input_shape=(13,2,1)))
model.add(MaxPooling2D(padding="same"))

model.add(Conv2D(128,kernel_size=5,strides=1,padding="same",activation="relu"))
model.add(MaxPooling2D(padding="same"))
model.add(Dropout(0.2))

model.add(Conv2D(256,kernel_size=5,strides=1,padding="same",activation="relu"))
model.add(MaxPooling2D(padding="same"))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(512,activation="relu"))
# model.add(Dropout(0.35))

model.add(Dense(10,activation="softmax"))


# In[138]:


#compiling
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# In[139]:


#training the model
model.fit(train_x,train_y,batch_size=64,epochs=100,validation_data=(test_x,test_y))


# In[ ]:




