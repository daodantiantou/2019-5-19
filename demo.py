import pandas as pd
data=pd.read_csv('./house_price.csv')
data1=data.dropna()
pd.set_option('display.max_columns',None)
data2=pd.get_dummies(data1[['dist','floor']])
data3=data2.drop(['dist_shijingshan','floor_high'],axis=1)
data4=pd.concat([data3,data1[['roomnum','halls','AREA','subway','school','price']]],axis=1)
# print(data4)
x=data4.iloc[:,:-1]
y=data4.iloc[:,-1:]
# print(y)
from sklearn import linear_model
from sklearn.model_selection import train_test_split
x_train,x_text,y_train,y_text=train_test_split(x,y,test_size=0.3,random_state=42)
model=linear_model.LinearRegression().fit(x_train,y_train)
result=model.predict([[0,0,0,0,0,0,0,3,2,60,1,1]])
score=model.score(x_text,y_text)
print(model.coef_)
print(model.intercept_)



