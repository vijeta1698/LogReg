
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from flask import Flask,request,render_template

app = Flask(__name__)

data = sm.datasets.fair.load_pandas().data
df = data.copy()
df['affair'] = (df.affairs>0).astype(int)
df.drop(columns = ['affairs'],inplace=True)
df = pd.get_dummies(columns = ['occupation','occupation_husb'],data=df)
y,X = df['affair'],df.drop(columns = ['affair'])
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 300)
clf = LogisticRegression()
clf.fit(x_train,y_train)

@app.route('/',methods = ['GET','POST'])
def ind():
    return render_template('index.html')
@app.route('/predict',methods =['GET','POST'])
def predict():
    rate_marriage = request.form['txtRm']
    age= request.form['txtAge']
    yrs_married= request.form['txtYrs_married']
    children = request.form['txtChildren']
    religious = request.form['txtReligious']
    educ = request.form['txtEduc']
    occupation_1 =  request.form['Occupation1.0']
    occupation_2 = request.form['Occupation2.0']
    occupation_3  = request.form['Occupation3.0']
    occupation_4  = request.form['Occupation4.0']
    occupation_5 =  request.form['Occupation5.0']
    occupation_6 =  request.form['Occupation6.0']
    occupation_husb_1 = request.form['OccupationHub1.0']
    occupation_husb_2=  request.form['OccupationHub2.0']
    occupation_husb_3 =  request.form['OccupationHub3.0']
    occupation_husb_4 = request.form['OccupationHub4.0']
    occupation_husb_5=  request.form['OccupationHub5.0']
    occupation_husb_6= request.form['OccupationHub6.0']
    predict =  clf.predict([[rate_marriage,age,yrs_married,children,religious,educ,occupation_1,occupation_2,occupation_3,occupation_4,
                  occupation_5,occupation_6,occupation_husb_1,occupation_husb_2,occupation_husb_3,
                  occupation_husb_4,occupation_husb_5,occupation_husb_6]])
    return render_template('index.html', prediction_text='Extramarital affair possibility -- {}'.format(predict[0]))


if __name__ == '__main__':
    app.run()