import pandas as pd 

df = pd.read_csv('dataset/heart.csv')   #membaca file dataset

from sklearn.neighbors import KNeighborsClassifier 
x = df.iloc[:, 0:13]    #memilih data kolom index 1 - 4 dari dataset sebagai data training
y = df.iloc[:, 13]  #memilih data kolom index 0 dari dataset sebagai nilai target

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x,y)

datatest={
    'age' : [30],
    'sex' : [1],
    'cp' : [3],
    'trestbps' : [150],
    'chol' : [273],
    'fbs' : [0],
    'restecg' : [1],
    'thalach' : [148],
    'exang' : [1],
    'oldpeak' : [1.2],
    'slope' : [2],
    'ca' : [3],
    'thal' : [1]
}


test = pd.DataFrame(datatest)
result = knn.predict(test) 
if result == 1 :
    result = 'Resiko Tinggi'
else :
    result = 'Resiko Rendah'

print("Result : ", result)

y_pred = knn.predict(x)
prob = knn.predict_proba(x)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y,y_pred))
print(classification_report(y,y_pred))

