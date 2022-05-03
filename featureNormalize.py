import numpy as np

# Feature Normalize Function
"""
Feature Normalize Normalizes the features in X
Returns a normalized version of X where the mean feature of each feature is 0
and the Standard Deviation of each feature is 1
This is often a good preprocessing step to do when working with learning algorithms
"""
def featureNormalize(X):
  #X_norm = X
  #mean = np.zeros((1, n))
  #sigma = np.zeros((1, n))

  # Compute Mean of each feature
  mean = np.mean(X, axis=0)
  print("mean =", mean)

  # Compute Standard Deviation of each feature
  sigma = np.std(X, axis=0)
  print("Standard Deviation = ", sigma)

  X_norm = ( X - mean) / sigma
  print("x_norm = \n", X_norm[:5, :])

  return X_norm
