import joblib
import pandas as pd
model=joblib.load("my model.h5")
f=model.feature_names_in_
a=int(input("enter itching"))
b=int(input("enter shivering"))
c=int(input("enter skin_rash"))
d=int(input("enter chills"))
e=int(input("enter joint_pain"))
'''f=int(input("enter chills"))
g=int(input("enter joint_pain"))
h=int(input("enter stomach_pain"))

i=int(input("enter acidity"))
j=int(input("enter ulcers_on_tongue"))
k=int(input("enter muscle_wasting"))
l=int(input("enter vomiting"))
m=int(input("enter spotting_ urination"))'''
d1={"itching":a,"shivering":b,"skin_rash":c,"chills":d,"joint_pain":e}
d2=pd.DataFrame([d1])
d2=pd.get_dummies(d2)
d2=d2.reindex(columns=f,fill_value=0)
p=model.predict(d2)
print(p)

if p==1:
    print("prognosis")
else:
    print("not prognosis")
