
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("datasets/train.csv").to_numpy()
clf = DecisionTreeClassifier()

  #training dataset
xtrain=data[0:21000,1:]
train_lable=data[0:21000,0]

clf.fit(xtrain,train_lable)

#testing data
xtest = data[21000: ,1:]
actual_lable = data[21000: ,0]

d= xtest[5]
d.shape=(28,28)
pt.imshow(255-d,cmap='gray', interpolation= 'nearest')
print(clf.predict( [xtest[5]] ))
pt.show()

p= clf.predict(xtest)
count=0
for i in range(0,21000):
    count+=1 if p[i]== actual_lable[i] else 0
print("Accuracy=", (count/21000)*100)    
