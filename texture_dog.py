import cv2

# 滤波器组和滤波器银行
# 使用一组滤波器（如Laplacian of Gaussian，DoG）提取图像的多尺度纹理特征。

# 读取灰度图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 应用Laplacian of Gaussian滤波器
log_image = cv2.GaussianBlur(image, (3, 3), 0)
log_image = cv2.Laplacian(log_image, cv2.CV_64F)

# 显示结果
cv2.imshow('Laplacian of Gaussian', log_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
