import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error,r2_score
import joblib
d=pd.read_csv("dataset.csv")
x=d.drop(columns=['prognosis'])
y=d['prognosis']
x=pd.get_dummies(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
model4=LinearRegression()
model4.fit(x_train,y_train_encoded)
pr=model4.predict(x_test)
r2=r2_score(y_test_encoded, pr)
print(f"R-squared:{r2*100}")
'''joblib.dump(model4,"my model.h5")'''
