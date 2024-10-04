#!/usr/bin/env python
# coding: utf-8

# # PCA -  Principal Component Analysis

# In[1]:



import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "pca_iris.data"
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

df


# In[2]:


features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = df[features]

y = df[['target']]



# In[3]:


x = StandardScaler().fit_transform(x)


# In[4]:


x


## PCA Projection 4 shape to 2 shape 

# In[5]:


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])


# In[6]:


principalDf



# In[7]:


final_dataframe = pd.concat([principalDf, df[['target']]], axis = 1)


# In[8]:


final_dataframe.head()


# In[ ]:







# In[9]:


dfsetosa= final_dataframe[df.target=='Iris-setosa']
dfvirginica = final_dataframe[df.target=='Iris-virginica']
dfversicolor = final_dataframe[df.target=='Iris-versicolor']
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')

plt.scatter(dfsetosa['principal component 1'], dfsetosa['principal component 2'],color='green')
plt.scatter(dfvirginica['principal component 1'], dfvirginica['principal component 2'],color='red')
plt.scatter(dfversicolor['principal component 1'], dfversicolor['principal component 2'],color='blue')



# In[10]:


targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['g', 'b', 'r']

plt.xlabel('principal component 1')
plt.ylabel('principal component 2')

for target, col in zip(targets,colors):
    dftemp = final_dataframe[df.target==target]
    plt.scatter(dftemp['principal component 1'], dftemp['principal component 2'], color=col)


# In[ ]:





# In[11]:


pca.explained_variance_ratio_


# In[12]:


pca.explained_variance_ratio_.sum()


# In[ ]:




