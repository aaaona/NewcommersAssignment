import cv2
import matplotlib.pyplot as plt

# 画像を読み込む
image1 = cv2.imread('C:/Users/aotec/OneDrive/idslab/NewcomersAssignment/pic4.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('C:/Users/aotec/OneDrive/idslab/NewcomersAssignment/pic6.jpg', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('C:/Users/aotec/OneDrive/idslab/NewcomersAssignment/pic7.jpg', cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread('C:/Users/aotec/OneDrive/idslab/NewcomersAssignment/pic9.jpg', cv2.IMREAD_GRAYSCALE)

# ヒストグラムの計算
hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([image3], [0], None, [256], [0, 256])
hist4 = cv2.calcHist([image4], [0], None, [256], [0, 256])

# ヒストグラムを比較表示
# 画像とヒストグラムを並べて表示
fig, axes = plt.subplots(4, 2, figsize=(10, 10))

images = [image1, image2, image3, image4]
hists = [hist1, hist2, hist3, hist4]
titles = ["Image 1", "Image 2", "Image 3", "Image 4"]

for i in range(4):
    # 画像表示
    axes[i][0].imshow(images[i], cmap='gray')
    axes[i][0].set_title(titles[i])
    axes[i][0].axis("off")

    # ヒストグラム表示
    axes[i][1].plot(hists[i], color='black')
    axes[i][1].set_xlim([0, 256])
    axes[i][1].set_title(f"Histogram {i+1}")

plt.tight_layout()
plt.show()