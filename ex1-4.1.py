import cv2

# 画像を読み込む
image1 = cv2.imread('C:/Users/aotec/OneDrive/idslab/NewcomersAssignment/pic4.jpg')
image2 = cv2.imread('C:/Users/aotec/OneDrive/idslab/NewcomersAssignment/pic5.jpg')
image3 = cv2.imread('C:/Users/aotec/OneDrive/idslab/NewcomersAssignment/pic8.jpg')

# SIFTの初期化（特徴点数を増やす調整）
sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)

# 特徴点を検出
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
keypoints3, descriptors3 = sift.detectAndCompute(image3, None)

bf = cv2.BFMatcher()

# image1 と image2 のマッチング
matches12 = bf.knnMatch(descriptors1, descriptors2, k=2)
good12 = []
for match1, match2 in matches12:
    if match1.distance < 0.75 * match2.distance:
        good12.append([match1])

sift_matches12 = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, good12, None, flags=2)

# image1 と image3 のマッチング
matches13 = bf.knnMatch(descriptors1, descriptors3, k=2)
good13 = []
for match1, match2 in matches13:
    if match1.distance < 0.75 * match2.distance:
        good13.append([match1])

sift_matches13 = cv2.drawMatchesKnn(image1, keypoints1, image3, keypoints3, good13, None, flags=2)

# 特徴点を描画
output1 = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0))
output2 = cv2.drawKeypoints(image2, keypoints2, None, color=(0, 255, 0))
output3 = cv2.drawKeypoints(image3, keypoints3, None, color=(0, 255, 0))

# 表示
cv2.imshow("SIFT Keypoints1", output1)
cv2.imshow("SIFT Keypoints2", output2)
cv2.imshow("SIFT Keypoints3", output3)
cv2.imshow("SIFT Matching Image1 and Image2", sift_matches12)
cv2.imshow("SIFT Matching Image1 and Image3", sift_matches13)

cv2.waitKey(0)
cv2.destroyAllWindows()
