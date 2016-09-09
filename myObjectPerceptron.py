import numpy as np

class Perceptron(object):

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
      for xi, yi in zip(x,y):
        update = yi - self.predict(xi)
        if(update != 0):
          tmpError += 1
          self._bias += update*self._etha
          self._weights += update*self._etha*xi
      self._errors.append(tmpError)


  def predict(self, x):
    y = np.dot(x,self._weights) + self._bias
    return np.where(y<0,-1,1)    

