import cv2

image1 = cv2.imread('C:/Users/aotec/OneDrive/idslab/NewcomersAssignment/pic1.jpeg') #元画像
image2 = cv2.resize(image1, None, fx=2, fy=2) #拡大画像
image3 = cv2.resize(image1, None, fx=1/2, fy=1/2) #縮小画像
image4 = cv2.rotate(image1,cv2.ROTATE_180) #180°回転
image5 = cv2.rotate(image1,cv2.ROTATE_90_CLOCKWISE) #90°時計回り回転
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) #グレースケール化
ret, image6 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) #二値化

print("元画像のサイズ:", image1.shape)
print("拡大後のサイズ:", image2.shape)
print("縮小後のサイズ:", image3.shape)

cv2.imshow('Original', image1)
cv2.imshow('Enlarged', image2)
cv2.imshow('Reduced', image3)
cv2.imshow('Rotate_180', image4)
cv2.imshow('Rotate_90_Clockwise',image5)
cv2.imshow('Binarization', image6)
cv2.waitKey(0)
cv2.destroyAllWindows()