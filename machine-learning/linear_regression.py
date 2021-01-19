# ==============================
# LINEAR REGRESSION
# ==============================
# Our model takes the form y = wx + b + e, where:
#   y   dependent variable
#   w   variable coefficients
#   x   independent variables
#   b   intercept (constant) term
#   e   error term assumed to follow ~ N(0, s)

# A variety of conditions must be met for this to be an appropriate modeling
# technique. This script assumes these to hold true and instead focuses on the
# algorithmetic implementation.

# ==============================
# SETUP
# ==============================
import numpy as np   # Multi-dimensional arrays
import pandas as pd  # Reading in dataset
from os import path

# Read & format data
dataset = pd.read_csv(path.join(".", "datasets", "Mall_Customers.csv"))

# Independent (predictor) variables
X = dataset[["Male", "Age", "Annual_Income"]].values

# M = # of rows, N = # of features.
M, N = X.shape

# Dependent (predicted) variable, reshaped to an Mx1 matrix.
Y = np.reshape(dataset.Spending_Score.values, (M, 1))

# With parameters w and b, we can compute the predictions from linear
# regression as:
def linear(w, b, x):
    return np.matmul(x, w) + b

# ==============================
# ANALYTIC SOLUTION 
# ==============================
# This approach solves the solution in a single step. We assume that the
# first variable in the X matrix is a vector of all 1's, which ultimately will
# represent the intercept `b` term. We'll make this adjustment first.
X_i = np.concatenate((np.ones((M, 1)), X), axis=1)

# Our model coefficients, theta, can then be calculated via normal equations:
#   X theta = Y
#   Xt X theta = Xt Y 
#   theta = (Xt X)**-1 Xt Y 
X_t = np.transpose(X_i)
theta_analytic = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_t, X_i)), X_t), Y)
w_analytic = theta_analytic[1:, 0].reshape((N, 1))
b_analytic = theta_analytic[0, 0]

print(w_analytic)
print(b_analytic)
# print(linear(w_analytic, b_analytic, X))

