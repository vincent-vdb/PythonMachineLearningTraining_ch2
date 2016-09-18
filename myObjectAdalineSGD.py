import numpy as np

class AdalineSGD(object):

  def __init__(self, _etha, _nEpochs):#, _bias, _weights):
    self._etha = _etha
    self._nEpochs = _nEpochs
#    self._bias = _bias
#    self._weights = _weights
    self._errors = []


  def fit(self, x, y):
    self._bias = 0
    self._weights = np.zeros(x.shape[1])

    for iEpochs in range(0,self._nEpochs):

      random = np.random.permutation(len(y))
      randomX, randomY = x[random], y[random]
      tmpSumCost = 0
      for xi, yi in zip(randomX, randomY):
        tmperrors = yi - self.activation(xi)
        self._bias += self._etha*tmperrors
        self._weights += self._etha*np.dot(xi,tmperrors)
        tmpCost = 0.5*np.square(tmperrors)
        tmpSumCost += tmpCost/len(y)
      self._errors.append(tmpSumCost)




  def predict(self, x):
    y = np.dot(x,self._weights) + self._bias
    return np.where(y<0,-1,1)    

  def costFunction(self, x, y):
    phi = self.activation(x)
    return 0.5*np.sum(np.square(y - phi))

  def costFunctionPrime(self,x,y):
    phi = np.dot(x, self._weights) + self._bias
    return -x*np.sum(y-phi)

  def activation(self,x):
    return np.dot(x, self._weights) + self._bias

