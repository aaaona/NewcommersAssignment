import cv2

image1 = cv2.imread('C:/Users/aotec/OneDrive/idslab/NewcomersAssignment/pic2.jpg')
image2 = cv2.imread('C:/Users/aotec/OneDrive/idslab/NewcomersAssignment/pic3.jpg')

diff = cv2.absdiff(image1, image2) #差分を計算

cv2.imshow('Original1', image1)
cv2.imshow('Original2', image2)
cv2.imshow('Difference', diff)
cv2.waitKey(0)
cv2.destroyAllWindows()