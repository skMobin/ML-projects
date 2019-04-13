#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


fashion_train = pd.read_csv('D:/GH/fashion-mnist_train.csv',sep = ',')


# In[3]:


fashion_test = pd.read_csv('D:/GH/fashion-mnist_test.csv',sep = ',')


# In[4]:


fashion_test.shape


# In[5]:


training = np.array(fashion_train)


# In[6]:


testing= np.array(fashion_test)


# In[7]:


testing


# In[8]:


import random
i= random.randint(1,40000)
img = cv2.Sobel(training[i,1:],cv2.CV_64F,0,1,ksize=1)
plt.imshow(img.reshape(28,28))
dict1 = {0:"T-shirt/top",1:"Trouser",2:"Pullover",3:"Dress",4:"Coat",5:"Sandal",6:"Shirt",7:"Sneaker",8:"Bag",9:"Ankle boot"}
label = training[i,0]
dict1[label]


# In[ ]:


# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot


# In[9]:


X_train = training[:,1:]
y_train = training[:,0]


# In[10]:


X_test = testing[:,1:]
y_test = testing[:,0]


# In[11]:




# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train,X_validate,y_train,y_validate = train_test_split(X_train, y_train,test_size = 0.2, random_state = 12345)


# In[14]:


X_train = X_train.reshape(X_train.shape[0], *(28,28,1))
X_test = X_test.reshape(X_test.shape[0],*(28,28,1))
X_validate = X_validate.reshape(X_validate.shape[0],*(28,28,1))


# In[15]:



import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


# In[16]:


cnn_model = Sequential()


# In[17]:


cnn_model.add(Conv2D(32,3,3, input_shape = (28,28,1),activation = 'relu'))


# In[18]:


cnn_model.add(MaxPooling2D(pool_size=(2,2)))


# In[19]:


cnn_model.add(Flatten())


# In[20]:


cnn_model.add(Dense(output_dim = 32, activation = 'relu' ))


# In[21]:


cnn_model.add(Dense(output_dim = 10, activation = 'sigmoid' ))


# In[22]:


cnn_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(lr=0.001),metrics =['accuracy'])


# In[23]:


cnn_model.fit(
    X_train, y_train, batch_size=128,
    epochs=10, verbose=1,
    validation_data=(X_validate, y_validate))


# In[24]:


evaluation = cnn_model.evaluate(X_test, y_test)
print('{: .3f}'.format(evaluation[1]))


# In[25]:


predicted_classes = cnn_model.predict_classes(X_test)


# In[26]:


predicted_classes


# In[27]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predicted_classes)
plt.figure(figsize = (14,10))
sns.heatmap(cm, annot= True)


# In[31]:


import numpy as np
import cv2
image = cv2.imread(r'D:\SM\MP\project\tshirt-img\pullover1.jpg', 0)
res = cv2.resize(image, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
res=res.reshape(1,28,28,1)


# In[32]:


cnn_model.predict_classes(res)


# In[30]:


import pickle
trained_model = open('model.pickle','wb')
pickle.dump(cnn_model.predict_classes(),trained_model)
trained_model.close()
