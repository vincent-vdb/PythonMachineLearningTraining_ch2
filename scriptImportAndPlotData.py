import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Perceptron import *


#import data from database
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail


#put data to x and y vectors
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)
x = df.iloc[0:100,[0,2]].values

#plot data
plt.scatter(x[:50,0], x[:50,1], color='red',marker='o',label='setosa')
plt.scatter(x[50:100,0], x[50:100,1], color='blue',marker='x',label='versicolor')

plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()

print("before perceptron instantiation")

ppn = Perceptron(eta=0.1, n_iter=10)
print("before perceptron fit")
ppn.fit(x,y)

plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')

plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()


