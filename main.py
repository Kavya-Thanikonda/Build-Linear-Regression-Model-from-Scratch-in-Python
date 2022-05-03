# Building Linear Regression Modek from Scratch in Python
print("Building Linear Regression Modek from Scratch in Python")

# ======================= Step 1: Importing Libraries ==========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Functions from other files
from featurenormalize import featureNormalize
from computecost import computeCost
from gradientdescent import gradientDescent

# ======================= Step 2: Loading the Data =============================
print("\nLoading the Dataset...")

dataset = np.loadtxt('ex1data2.txt', delimiter=',')
#print(dataset[:5, :])

X = dataset[:, :-1]
y = dataset[:, -1]

# Print out some data points
print("First 5 Examples from the dataset: ")
print("x = \n", X[:5, :])
print("y = ", y[:5])

m = len(y)
print("\nNumber of Training Examples: ", m)
n = len(X[1])
print("Number of Features: ", n)

# ======================= Step 3: Feature Normalization ========================
# Scale Features and set them to zero mean
print("\n\nNormalizing Features ... ")
X = featureNormalize(X)


# Add intercept term to X
print("\n\nAdding intercept term to X")
x_0 = np.ones((m,1))
X = np.concatenate((x_0, X), axis=1)
print("x = \n", X[:5, :])


# ======================= Step 4: Computing Initial Cost =======================
# Initiate theta and compute initial Cost
theta = np.zeros((n+1, 1))
print("Initializing theta = \n", theta)

print("\n\nComputing Initial Cost...")
# Save the Cost J in every iteration
J = computeCost(X, y, theta)

print("Initial Cost: ", J)


# ======================= Part 3: Gradient Descent =============================
print("\n\nRunning Gradient Descent...")

# Choose some value of alpha
alpha = 0.01
num_iters = 400

theta, J_history = gradientDescent(X, y, alpha, num_iters)

# Display Gradient Descent's results
print("\nTheta computed from gradient descent: \n", theta)


# ======================= Step 5: Plot the Convergence =========================
print("\n\n Convergence of Gradient Descent")
plt.plot(list(range(0,400)), J_history[0:400])
plt.xlabel("number of iterations")
plt.ylabel("Cost")
plt.show()

# ======================= Step 6: Predict ======================================
# Estimate the price of a 1650 sq-ft, 3 br house
X_test = np.array([1, 1650, 3]).reshape(1,3)
price = np.matmul(X_test, theta)
print("\n\nPredicted price of a 1650 sq-ft, 3 br house (using gradient descent): ", price[0][0])
