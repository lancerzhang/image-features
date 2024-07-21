import cv2
from skimage.feature import graycomatrix, graycoprops

# 灰度共生矩阵（GLCM）
# 灰度共生矩阵是一种统计方法，用于描述图像中像素对的灰度值共现情况，从而提取纹理特征。
# 灰度共生矩阵用于提取图像中的统计特征，如对比度、能量、同质性等。

# 读取图像并转换为灰度图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 计算灰度共生矩阵
glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

# 提取纹理特征
contrast = graycoprops(glcm, 'contrast')
dissimilarity = graycoprops(glcm, 'dissimilarity')
homogeneity = graycoprops(glcm, 'homogeneity')
energy = graycoprops(glcm, 'energy')
correlation = graycoprops(glcm, 'correlation')
ASM = graycoprops(glcm, 'ASM')

print("Contrast: ", contrast)
print("Dissimilarity: ", dissimilarity)
print("Homogeneity: ", homogeneity)
print("Energy: ", energy)
print("Correlation: ", correlation)
print("ASM: ", ASM)
