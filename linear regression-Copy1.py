import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")
df

plt.xlabel('area(sqr ft)')
plt.ylabel('Price(US$)')
plt.scatter(df.area, df.price, color="red", marker="*")
plt.show()


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area(sqr ft)')
plt.ylabel('Price(US$)')
plt.scatter(df.area, df.price, color="red", marker="*")
plt.show()


reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

predicted_price = reg.predict([[3300]])
print(predicted_price)

reg.coef_
reg.intercept_

135.78767123*3300+180616.43835616432

d= pd.read_csv("areas.csv")
d.head()

p=reg.predict(d)
p

d['price']=p
d

d.to_csv("prediction.csv",index=False)

plt.xlabel('area(sqr ft)')
plt.ylabel('Price(US$)')
plt.scatter(df.area, df.price, color="red", marker="*")
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.show()
