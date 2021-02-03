import numpy as np

import pandas as pd

from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
plt.style.use('fivethirtyeight')



#df = pd.read_csv('DrDoS_DNS_upgraded2.csv')
#df = pd.read_csv('MSSQL.csv')
#df = pd.read_csv('NetBIOS.csv')
df=pd.read_csv('mosaic3.csv')
#df=pd.read_csv('CICID1.csv')
#df = df.convert_objects(convert_numeric=True)
#df.drop_duplicates(inplace = True)
df.replace([np.inf, -np.inf], np.nan)



X=df.iloc[:, :-1 ].to_numpy()
imputer= Imputer(missing_values="NaN" , strategy="mean" ,axis=0)
imputer = imputer.fit(X[: ,0:4])
X[: ,0:4] = imputer.transform(X[: ,0:4])

# Get all of the rows from the last column
y = df.iloc[:,-1].to_numpy()
#print(sum(y))
'''
from sklearn.preprocessing import LabelEncoder
labelencoder= LabelEncoder()
y = labelencoder.fit_transform(y)
'''

for i in range(len(y)):
    if y[i]=="BENIGN":
        y[i]=0
    else:
        y[i]=1
        
from sklearn.preprocessing import LabelEncoder
labelencoder= LabelEncoder()
y = labelencoder.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 5)

from sklearn.preprocessing import StandardScaler
scale_X= StandardScaler()
X_train= scale_X.fit_transform(X_train)
X_test= scale_X.transform(X_test)


#random forest:
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)
model.score(X_test, y_test)

#svm:
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)





y_predicted = model.predict(X_test)
y_predicted = [1 if y>=0.5 else 0 for y in y_predicted]
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
#cm = confusion_matrix(y_test, y_predicted)
#Threshold
print(classification_report(y_test ,y_predicted ))
print('Confusion Matrix: \n',confusion_matrix(y_test,y_predicted))
print()
print('Accuracy: ', accuracy_score(y_test,y_predicted))
print()



