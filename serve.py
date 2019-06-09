from flask import Flask,render_template,request

app=Flask(__name__)

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
model1=linear_model.LogisticRegression(max_iter=10000,tol=0.00001,solver='liblinear').fit(x,y)

@app.route('/')
def fangjia():
    return render_template('login.html')

@app.route('/login',methods=['POST'])
def login():
    sex=int(request.form['sex'])
    age=int(request.form['age'])
    salary=int(request.form['salary'])
    res=model1.predict([[sex,age,salary]])[0]
    if res:
        return render_template('fangjia.html')
    else:
        return render_template('index.html')

@app.route('/yuce',methods=['POST'])
def yuce():
    arr=[0,0,0,0,0,0,0,0,0,0,0,0]
    dist=int(request.form['dist'])
    if dist== -1:
        pass
    else:
        arr[dist]=1
    flool=int(request.form['floor'])
    if flool==-1:
        pass
    else:
        arr[flool] = 1
    roomnum=int(request.form['roomnum'])
    arr[7]=roomnum
    halls = int(request.form['halls'])
    arr[8] = halls
    area = int(request.form['area'])
    if area:
        arr[9] = area
    else:
        arr[9]=1

    subway = int(request.form['subway'])
    arr[10] = subway
    school = int(request.form['school'])
    arr[11] = school
    res=model.predict([arr])[0][0]
    print(res)
    return render_template('fangjia.html',data={'res':res})
app.run()