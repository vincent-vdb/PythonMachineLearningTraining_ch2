import numpy as np

class AdalineGD(object):

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
      tmpError = 0
#      for xi, yi in zip(x,y):
#      update = costFunctionPrime(x,y)
      self._bias += self._etha*np.sum(y-self.activation(x))


#      self._weights += self._etha*np.dot(np.sum(y-self.activation(x)),x.T)
      self._weights += self._etha*np.dot(x.T,(y-self.activation(x)))

#        update = yi - self.predict(xi)
#        if(update != 0):
#          tmpError += 1
#          self._bias += update*self._etha
#          self._weights += update*self._etha*xi
      print(self.costFunction(x,y))
      self._errors.append(self.costFunction(x,y))


  def predict(self, x):
    y = np.dot(x,self._weights) + self._bias
    return np.where(y<0,-1,1)    

  def costFunction(self, x, y):
    phi = np.dot(x, self._weights) + self._bias
    j = np.sum(np.square(y - phi))
    return j

  def costFunctionPrime(self,x,y):
    phi = np.dot(x, self._weights) + self._bias
    return -x*np.sum(y-phi)

  def activation(self,x):
    return np.dot(x, self._weights) + self._bias

