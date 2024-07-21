import cv2
import pywt

# Haar小波变换
# Haar小波变换是一种用于多尺度分析的工具，可以提取图像中不同尺度和方向的纹理信息。

# 读取图像并转换为灰度图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 进行Haar小波变换
coeffs2 = pywt.dwt2(image, 'haar')
LL, (LH, HL, HH) = coeffs2

# 显示变换后的图像
cv2.imshow('Approximation', LL)
cv2.imshow('Horizontal detail', LH)
cv2.imshow('Vertical detail', HL)
cv2.imshow('Diagonal detail', HH)
cv2.waitKey(0)
cv2.destroyAllWindows()
