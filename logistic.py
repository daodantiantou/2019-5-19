import pandas as pd
ldata=pd.read_csv('./logistic_dat.csv')
ldata1=ldata.dropna()
ldata2=pd.get_dummies(ldata1.Gender)
ldata3=ldata2.drop('Female',axis=1)
# print(ldata1)
ldata4=pd.concat([ldata3,ldata1[['Age','EstimatedSalary','Purchased']]],axis=1)
print(ldata4)
x=ldata4.iloc[:,:-1]
y=ldata4['Purchased']
from sklearn import linear_model
model=linear_model.LogisticRegression(max_iter=10000,tol=0.00001,solver='liblinear').fit(x,y)
res=model.predict([[1,4,1]])
score=model.score(x,y)
print(res)
print(score)