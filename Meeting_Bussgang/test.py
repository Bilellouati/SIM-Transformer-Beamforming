# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:07:41 2024

@author: AM271697
"""

import numpy as np

def element_wise_correlation(X, Y):
    # Check if lengths of vectors are the same
    if len(X) != len(Y):
        raise ValueError("Vectors must have the same length")

    # Calculate correlation coefficient for each pair of elements
    correlations = []
    for i in range(len(X)):
        correlation_coefficient = np.corrcoef([X[i]], [Y[i]])[0, 1]
        correlations.append(correlation_coefficient)
    
    return correlations

# Example vectors X and Y
X = [1, 2, 3, 4, 5]
Y = [2, 4, 6, 8, 10]

# Calculate correlation element by element
correlations = element_wise_correlation(X, Y)
print("Correlation coefficients (element by element):", correlations)