import numpy as np

# Gradient Descent Function
"""
Gradient Descent Function Performs Gradient Descent to learn the theta
"""
def gradientDescent(X, y, theta, alpha, num_iters):
  m = len(y)
  y = y.reshape(len(y), 1)

  #print("theta.shape", theta.shape)
  #print("X.shape", X.shape)
  #print("y.shape", y.shape)

  J_history = np.zeros((num_iters, 1))

  for i in range(num_iters):
    #hypothesis = np.transpose(theta) * X
    #hypothesis = X * theta
    hypothesis = np.matmul(X, theta)

    delta = (hypothesis - y) * X
    delta = np.sum(delta, axis=0)
    delta = delta.reshape(len(delta), 1)

    theta = theta - alpha * (1/m) * delta

    # Save the Cost J in every iteration
    J_history[i] = computeCost(X, y, theta)

  return theta, J_history
