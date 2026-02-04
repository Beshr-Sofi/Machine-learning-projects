import pandas as pd
import numpy as np

class LogisticRegression:
    """
    Logistic Regression classifier with L2 regularization.
    
    Parameters:
    -----------
    learning_rate: Magnitude of the steps taken during optimization.
    lambda_reg: Regularization strength (higher means more penalty on weights).
    num_iterations: Number of passes over the training dataset.
    """
    def __init__(self, learning_rate=0.01, lambda_reg=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        # Maps inputs to a value between 0 and 1
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, x, y, weights, bias):
        """
        Calculates the Binary Cross-Entropy loss with L2 regularization.
        """
        m = len(y)
        # Vectorized implementation:
        z = np.dot(x, weights) + bias
        a = self.sigmoid(z)
        
        # Log Loss + L2 Penalty
        cost = (-1/m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
        reg_cost = (self.lambda_reg / (2 * m)) * np.sum(np.square(weights))
        
        return cost + reg_cost
    
    def compute_gradients(self, x, y, weights, bias):
        """
        Computes the partial derivatives of the cost function w.r.t weights and bias.
        """
        m = x.shape[0]
        z = np.dot(x, weights) + bias
        a = self.sigmoid(z)
        
        # Errors (predictions - actual labels)
        dz = a - y 
        
        # Gradient of weights (including L2 derivative)
        dw = (1 / m) * np.dot(x.T, dz) + (self.lambda_reg / m) * weights
        # Gradient of bias
        db = (1 / m) * np.sum(dz)
        
        return dw, db
    
    def gradient_descent(self, x, y, weights, bias):
        """
        Updates weights and bias iteratively to minimize the cost.
        """
        for i in range(self.num_iterations):
            dw, db = self.compute_gradients(x, y, weights, bias)
            
            # Update parameters
            weights -= self.learning_rate * dw
            bias -= self.learning_rate * db
            
            # Print cost every 100 iterations to monitor progress
            if i % 100 == 0:
                cost = self.compute_cost(x, y, weights, bias)
                print(f"Cost after iteration {i}: {cost:.4f}")
                
        return weights, bias
        
    def fit(self, x, y):
        """
        Trains the model on features x and target y.
        """
        n_features = x.shape[1]
        # Initialize weights as zeros
        weights = np.zeros(n_features)
        bias = 0
        
        self.weights, self.bias = self.gradient_descent(x, y, weights, bias)

    def predict(self, x):
        """
        Predicts class labels (0 or 1) for given input.
        """
        z = np.dot(x, self.weights) + self.bias
        probabilities = self.sigmoid(z)
        # Threshold at 0.5: if prob >= 0.5, class 1; else class 0
        return np.where(probabilities >= 0.5, 1, 0)



#---------------------
#BEFORE USING THE AI
#---------------------
# class LogisticRegression:
#     def __init__(self,learning_rate=0.01, lambda_reg=0.01, num_iterations=1000):
#         self.learning_rate = learning_rate
#         self.lambda_reg = lambda_reg
#         self.num_iterations = num_iterations
#         self.weights = None
#         self.bias = None
    
#     def sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))
    
#     def compute_cost(self, x, y, weights, bias):
#         m = len(y)
#         n = len(weights)
#         cost = 0
#         for i in range(m):
#             z = np.dot(weights, x[i]) + bias
#             a = self.sigmoid(z)
#             cost += -y[i] * np.log(a) - (1 - y[i]) * np.log(1 - a)
#         reg = 0
#         for j in range(n):
#             reg += weights[j] ** 2
#         cost = cost / (m) + (self.lambda_reg / (2 * m)) * reg
#         return cost
    
#     def compute_gradients(self, x, y, weights, bias):
#         m = len(y)
#         n = len(weights)
#         dw = np.zeros(n)
#         db = 0
#         for i in range(m):
#             z = np.dot(weights, x[i]) + bias
#             a = self.sigmoid(z)
#             for j in range(n):
#                 dw[j] += (a - y[i]) * x[i][j]
#             db += (a - y[i])
#         dw = dw / m + (self.lambda_reg / m) * weights
#         db = db / m
#         return dw, db
    
#     def gradient_descent(self, x, y, weights, bias):
#         for i in range(self.num_iterations):
#             dw, db = self.compute_gradients(x, y, weights, bias)
#             weights -= self.learning_rate * dw
#             if i % 100 == 0:
#                 print(f"Cost after iteration {i}: {self.compute_cost(x, y, weights, bias)}")
#             bias -= self.learning_rate * db
#         return weights, bias
        
#     def fit(self, x, y):
#         n = x.shape[1]
#         weights = np.zeros(n)
#         bias = 0
#         self.weights, self.bias = self.gradient_descent(x, y, weights, bias)

#     def predict(self, x):
#         z = np.dot(x, self.weights) + self.bias
#         a = self.sigmoid(z)
#         y_pred = np.where(a >= 0.5, 1, 0)
#         return y_pred
    
