from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add', methods=['POST'])
def add():
    try:
        import joblib
        import pandas as pd
        rfc=joblib.load("my model.h5")
        f=rfc.feature_names_in_
        a = int(request.form['a'])
        b= int(request.form['b'])
        c = int(request.form['c'])
        t = int(request.form['t'])
        e = int(request.form['e'])
        newdata={'itching':a,'shivering':b,'skin_rash':c,'chills':t,'joint_pain':e}
        newdata=pd.DataFrame([newdata])
        newdata=pd.get_dummies(newdata)
        newdata=newdata.reindex(columns=f,fill_value=0)
        result = rfc.predict(newdata)
    except ValueError:
        result = "Invalid input! Please enter numbers only."
    
    return redirect(url_for('result', result=result))

@app.route('/result')
def result():
    result = request.args.get('result')
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
