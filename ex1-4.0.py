import cv2

image1 = cv2.imread('C:/Users/aotec/OneDrive/idslab/NewcomersAssignment/pic4.jpg')
image2 = cv2.imread('C:/Users/aotec/OneDrive/idslab/NewcomersAssignment/pic5.jpg')
image3 = cv2.imread('C:/Users/aotec/OneDrive/idslab/NewcomersAssignment/pic8.jpg')
orb = cv2.ORB_create()

keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
keypoints3, descriptors3 = orb.detectAndCompute(image3, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# image1 と image2 のマッチング
matches12 = bf.match(descriptors1, descriptors2)
matches12 = sorted(matches12, key = lambda x:x.distance)
orb_matches12 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches12[:20], None, flags=2)

# image1 と image3 のマッチング
matches13 = bf.match(descriptors1, descriptors3)
matches13 = sorted(matches13, key = lambda x:x.distance)
orb_matches13 = cv2.drawMatches(image1, keypoints1, image3, keypoints3, matches13[:20], None, flags=2)

output1 = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0), flags=0)
output2 = cv2.drawKeypoints(image2, keypoints2, None, color=(0, 255, 0), flags=0)
output3 = cv2.drawKeypoints(image3, keypoints3, None, color=(0, 255, 0), flags=0)

# 表示
cv2.imshow("ORB Keypoints1", output1)
cv2.imshow("ORB Keypoints2", output2)
cv2.imshow("ORB Keypoints3", output3)
cv2.imshow("ORB Matching Image1 and Image2", orb_matches12)
cv2.imshow("ORB Matching Image1 and Image3", orb_matches13)

cv2.waitKey(0)
cv2.destroyAllWindows()
