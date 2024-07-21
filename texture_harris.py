import cv2

# 哈里斯角点检测
# 哈里斯角点检测是一种用于检测图像中角点（或纹理特征）的算法。

# 读取灰度图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 检测角点
dst = cv2.cornerHarris(image, 2, 3, 0.04)

# 增强角点
dst = cv2.dilate(dst, None)

# 标记角点
image[dst > 0.01 * dst.max()] = 255

# 显示图像
cv2.imshow('Harris Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
