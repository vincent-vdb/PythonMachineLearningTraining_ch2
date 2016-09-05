import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail

import matplotlib.pyplot as plt
import numpy as np

y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)
x = df.iloc[0:100,[0,2]].values

plt.scatter(x[:50,0], x[:50,1], color='red',marker='o',label='setosa')
plt.scatter(x[50:100,0], x[50:100,1], color='blue',marker='x',label='versicolor')

plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()



