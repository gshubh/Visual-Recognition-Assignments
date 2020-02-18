import numpy as np
import cv2
import colorsys
import math

img = cv2.imread('/home/iiitb/Desktop/watch.jpg')
X = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# def flatten_matrix(matrix):
#     vector = matrix.flatten(1)
#     vector = vector.reshape(1, len(vector))
#     return vector



def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T) #ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)   #Data whitening






#X -= np.mean(X, axis = 0) # zero-center the data (important)
# cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix

# U,S,V = np.linalg.svd(cov)

# Xrot = np.dot(X, U) # decorrelate the data

# #Xrot_reduced = np.dot(X, U[:,:200]) # Xrot_reduced becomes [N x 100]

# # whiten the data:
# # divide by the eigenvalues (which are square roots of the singular values)
# Xwhite = Xrot / np.sqrt(S + 1e-5)

Xwhite = zca_whitening(X)
cv2.imshow('Whitened Image', Xwhite)

cv2.waitKey(0)          
cv2.destroyAllWindows()
#print(matrix.shape)

