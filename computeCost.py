import numpy as np

# Compute Cost Function
def computeCost(X, y, theta):
  """
  Computes Cost of using theta as the parameter for linear regression to fit the data points in X and y
  """
  y = y.reshape(len(y), 1)

  hypothesis = np.matmul(X, theta)
  cost = 1/(2*m) * np.sum(np.power((hypothesis - y), 2))
  
  return cost
