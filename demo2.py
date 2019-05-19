import pandas as pd
data=pd.read_csv('./house_price1.csv')
x=data[['area']]
y=data.iloc[:,-1:]
from sklearn import linear_model
model=linear_model.LinearRegression().fit(x,y)
a=model.coef_
b=model.intercept_
import matplotlib.pyplot as plt
fig=plt.figure()
plt.scatter(x,y)
x1=x
y1=a*x+b
plt.plot(x1,y1)
plt.show()

