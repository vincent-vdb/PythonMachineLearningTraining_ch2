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
      tmperrors = y - self.activation(x)
      self._bias += self._etha*np.sum(tmperrors)
      self._weights += self._etha*np.dot(x.T,tmperrors)

      print(self.costFunction(x,y))
      self._errors.append(self.costFunction(x,y))


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

