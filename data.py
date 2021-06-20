import pandas as pd
import csv
import numpy as np 
import plotly.figure_factory as ff
import plotly.express as px 

df = pd.read_csv("data.csv")
toefl_score = df["TOEFL Score"].tolist() 
gre_score = df["GRE Score"].tolist()

fig = px.scatter(x =toefl_score , y = gre_score)
fig.show()

import plotly.graph_objects as go

results = df["results"].tolist()
colors =[]

for data in results :
    if data == 1 :
        colors.append("green")
    else :
        colors.append("red")


fig1  = go.Figure(
    data = go.Scatter(
        x = toefl_score,
        y = gre_score,
        mode = 'markers',
        marker = dict(color = colors)
       
    )
)

fig1.show()

hours = df[["toefl_score","gre_score"]]

results = df["results"]

from sklearn.model_selection import train_test_split

toefl_train,hours_test,results_train,results_test = train_test_split(toefl,results,test_size = 0.25,random_state = 0)
print(toefl_train)

from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0)
classifier.fit(toefl_train,results_train)

results_pred = classifier.predict(hours_test)
from sklearn.metrics import accuracy_score

print("accuracy: ", accuracy_score(results_test,results_pred))

from sklearn.preprocessing import StandardScaler
sc_x   = StandardScaler()

toefl_train = sc_x.fit_transform(toefl_train)

user_toefl_score = int(input("enter TOEFL score "))
user_gre_score = int(input("enter GRE score "))
user_test = sc_x.transform([[user_toefl_study , user_gre_slept]])

user_result_pred = classifier.predict(user_test)

if user_result_pred[0] == 1:
    print("This user may pass")
else :
    print("This user may not pass")