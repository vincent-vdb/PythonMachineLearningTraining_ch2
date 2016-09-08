import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#import data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

#extract needed data
y_raw=df.iloc[0:100,4].values
x=df.iloc[0:100,[0,2]].values

#convert raw flower name to values 1 and -1
y=np.where(y_raw=='Iris-setosa', -1,1)


#define the weights and the bias to zero
w = np.zeros(x.shape[1])
b = 0

#define the number of epochs
n_epochs = 10
#define the learning rate etha
etha = 0.1


#define the method that computes the output y of the perceptron for an input x
def perceptronFit(x,w,b):
  y = np.dot(w,x) + b
  if y > 0:
    return 1
  else :
    return -1



#define the number of errors vectors per epoch
errors = []

#update the perceptron with the data over n epochs
for i_epoch in range(0,n_epochs):
  #define the number of errors for this epoch
  error = 0
  #loop over the training values
  for xi, yi in zip(x,y):
#    print(xi)
#    print(w)
#    print(b)
    update = yi - perceptronFit(xi,w,b)

    if(update != 0) :
      w = w + update*etha*xi
      b = b + update*etha
      error = error + 1
  print(error)
  errors.append(error)


print(errors)

plt.plot(range(0,n_epochs),errors,marker='x')
plt.xlabel('iteration')
plt.ylabel('misclassified data')
plt.show()

