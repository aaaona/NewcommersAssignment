import cv2
import numpy as np
A = np.array([[1, 2],[3, 4]])
B = np.array([[5, 6],[7, 8]])
result = np.dot(A, B)
print('A:n', A)
print('B:', B)
print('A+B:\n', A+B)
print('A-B:\n', A-B)
print('A*B:\n', A*B)
print('A*B(行列積):\n', result)
print('A/B:\n', A/B)