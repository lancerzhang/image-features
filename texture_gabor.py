import cv2
import numpy as np

# Gabor滤波器
# Gabor滤波器是一种线性滤波器，可以提取图像的特定频率和方向的纹理特征。
# Gabor滤波器在频域中具有选择性，可以用来提取纹理特征。

# 读取图像并转换为灰度图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 定义Gabor滤波器参数
ksize = 31  # 滤波器尺寸
sigma = 4.0  # 高斯函数的标准差
theta = np.pi / 4  # 滤波器的方向
lambd = 10.0  # 波长
gamma = 0.5  # 空间纵横比
psi = 0  # 相位偏移

# 创建Gabor滤波器
gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)

# 应用Gabor滤波器
filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)

# 显示滤波后的图像
cv2.imshow('Gabor', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
