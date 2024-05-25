#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[3]:


df = pd.read_csv("homeprices.csv")
df


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area(sqr ft)')
plt.ylabel('Price(US$)')
plt.scatter(df.area, df.price, color="red", marker="*")
plt.show()


# In[37]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area(sqr ft)')
plt.ylabel('Price(US$)')
plt.scatter(df.area, df.price, color="red", marker="*")
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.show()


# In[18]:


reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)


# In[21]:


predicted_price = reg.predict([[3300]])
print(predicted_price)


# In[23]:


reg.coef_


# In[24]:


reg.intercept_


# In[25]:


135.78767123*3300+180616.43835616432


# In[27]:


d= pd.read_csv("areas.csv")
d.head()


# In[30]:


p=reg.predict(d)
p


# In[31]:


d['price']=p


# In[32]:


d


# In[36]:


d.to_csv("prediction.csv",index=False)


# In[ ]:


**LINEAR REGRESSION FOR CANDA_PER_CAPITA_INCOME**


# In[60]:


ds=pd.read_csv("canada_per_capita_income.csv")
ds.head()


# In[66]:


ds = ds.rename(columns={
    'year': 'Year', 
    'per capita income (US$)': 'PerCapitaIncome'
})
ds


# In[76]:


plt.xlabel('Year')
plt.ylabel('PerCapitaIncome(US$)')
plt.scatter(ds.Year,ds.PerCapitaIncome,color="red", marker="+")
plt.show()


# In[70]:


regg = linear_model.LinearRegression()
regg.fit(ds[['Year']],ds.PerCapitaIncome)


# In[72]:


regg.predict([[2020]])


# In[73]:


regg.coef_


# In[74]:


regg.intercept_


# In[75]:


828.46507522*2020-1632210.7578554575


# In[78]:


plt.xlabel('Year')
plt.ylabel('PerCapitaIncome(US$)')
plt.scatter(ds.Year,ds.PerCapitaIncome,color="red", marker="+")
plt.plot(ds.Year,regg.predict(ds[['Year']]),color='blue')
plt.show()

