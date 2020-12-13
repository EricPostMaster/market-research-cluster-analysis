#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


pip install factor_analyzer


# In[3]:


from factor_analyzer import FactorAnalyzer


# In[4]:


import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np


# In[5]:


df = pd.read_csv('C:/kmeans/UIGV1311956_OP0.txt', sep='\t')


# In[7]:


df.head(5)


# In[8]:


df_fct =  df.drop(['UID','Const'], axis = 1)


# In[7]:


df_fct.head()


# In[7]:


fa = FactorAnalyzer(n_factors=5,rotation='quartimax')


# In[8]:


fa.fit(df_fct)


# In[9]:


loadings = fa.loadings_


# In[10]:


ev, v = fa.get_eigenvalues()


# In[11]:


xvals = range(1, df_fct.shape[1]+1)


# In[21]:


plt.scatter(xvals,ev)
plt.plot(xvals, ev)
plt.title('Scree Plot')
plt.xlabel('Factor')
plt.ylabel('Eigenvalues')
plt.yticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
plt.grid()
plt.show()


# In[16]:


ev


# In[17]:


v


# In[12]:


eigen=pd.DataFrame(v,columns=['eigenvalues']) 


# In[13]:


ev


# In[14]:


num_of_facts=sum(i >= 1 for i in ev)


# In[15]:


fa = FactorAnalyzer(n_factors=num_of_facts,rotation='quartimax')


# In[16]:


fa.fit(df_fct)


# In[17]:


loadings = fa.loadings_


# In[18]:


len(loadings)


# In[24]:


pip install openpyxl


# In[26]:


import pandas as pd 
factor_scores = fa.transform(df_fct)
factor_scores = pd.DataFrame(factor_scores)
loadings = pd.DataFrame(loadings)
factor_scores.to_excel('C:/kmeans/factor_scores.xlsx')
loadings.to_excel('C:/kmeans/loadings.xlsx')


# In[49]:


loadings_abs=loadings.abs()
loadings[loadings.abs() >= .4].count()


# Below is for KMEANS Clustering

# In[38]:


from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
import numpy as np


# In[55]:


X=df_fct
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X)


# In[57]:


plt.scatter(X[:,0],X[:,36], c=kmeans.labels_, cmap='rainbow')


# In[56]:


clusters


# In[59]:


mglearn.plots.plot_dbscan()


# In[61]:


from sklearn.cluster import DBSCAN


# In[1]:


from scipy.stats import pearsonr
import numpy as np


# In[2]:


def pearson_affinity(M):
   return 1 - np.array([[pearsonr(a,b)[0] for a in M] for b in M])


# In[ ]:




