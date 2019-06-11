
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn import utils
excel_f1='da.xlsx'
dataset=pd.read_excel(excel_f1,sheet_name=0,index_col=0)

#print(dataset.head()) #displays first few values
dataset=dataset.astype('int')
#The X variable contains the first four columns of the dataset (i.e. attributes) while y contains the labels.
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
#print(X)
#print(y)

#To create training and test splits, execute the following script:
#The above script splits the dataset into 80% train data and 20% test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print(len(X_train),len(X_test))
print(len(y_train),len(y_test))

#The following script performs feature scaling:
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#The first step is to import the KNeighborsClassifier class from the sklearn.neighbors library.
# In the second line, this class is initialized with one parameter, i.e. n_neigbours.
# This is basically the value for the K.
# There is no ideal value for K and it is selected after testing and evaluation,
# however to start out, 5 seems to be the most commonly used value for KNN algorithm.

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('hi')
print(confusion_matrix(y_test, y_pred))
print('bye')
print(classification_report(y_test, y_pred))
error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
