
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries
# installed It is defined by the kaggle/python docker image:
# https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in


import pandas as pd   # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm
from sklearn.model_selection import train_test_split  # , GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter)
# will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:

# In[ ]:

# labeled_images = pd.read_csv('../input/train.csv')
# images = labeled_images.iloc[0:7000,1:]
# labels = labeled_images.iloc[0:7000,:1]
# train_images, test_images,train_labels, test_labels = train_test_split(
# images, labels, train_size=0.8, random_state=0)


# In[ ]:


# i=50
# img=train_images.iloc[i].as_matrix()
# img=img.reshape((28,28))
# plt.imshow(img,cmap='gray')
# plt.title(train_labels.iloc[i,0])


# In[ ]:


# test_images[test_images>0]=1
# train_images[train_images>0]=1
# img=train_images.iloc[i].as_matrix().reshape((28,28))
# plt.imshow(img,cmap='binary')
# plt.title(train_labels.iloc[i])


# In[ ]:


# params = {'C': [0.001, 0.01, 0.1, 1, 5],
#                'gamma': [0.001, 0.01, 0.1, 1, 10]}
# clf = GridSearchCV(svm.SVC(random_state=0),params, cv=5, n_jobs=-1)
# clf.fit(train_images, train_labels.values.ravel())
# clf.score(test_images,test_labels)


# In[ ]:


# clf.best_params_


# In[ ]:


labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[:, 1:]
labels = labeled_images.iloc[:, :1]
train_images, test_images, train_labels, test_labels = train_test_split(
                            images, labels, train_size=0.8, random_state=0)


# In[ ]:


# images.shape


# In[ ]:


# labels.shape


# In[ ]:


# shape_l = [train_images.shape, test_images.shape, train_labels.shape,
# test_labels.shape]


# In[ ]:


# shape_l


# In[ ]:


test_images[test_images > 0] = 1
train_images[train_images > 0] = 1
# img = train_images.iloc[i].as_matrix().reshape((28,28))
# plt.imshow(img, cmap='binary')
# plt.title(train_labels.iloc[i])


# In[ ]:


clf = svm.SVC(random_state=0, gamma=0.01, C=5)
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images, test_labels)


# In[ ]:


test_data = pd.read_csv('../input/test.csv')


# In[ ]:


# test_data.shape


# In[ ]:


test_data[test_data > 0] = 1
results = clf.predict(test_data)


df = pd.DataFrame(results)
df.index.name = 'ImageId'
df.index += 1
df.columns = ['Label']
df.to_csv('results.csv', header=True)
