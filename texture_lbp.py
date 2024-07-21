import cv2
from skimage.feature import local_binary_pattern

# 局部二值模式（LBP）
# LBP是一种用于纹理描述的强大工具，它通过比较每个像素与其邻域像素的灰度值，将结果二值化并编码为二进制数。
# 局部二值模式是一种用于纹理分类的简单有效的方法。

# 读取图像并转换为灰度图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 设置LBP参数
radius = 1  # 邻域半径
n_points = 8 * radius  # 邻域点数

# 计算LBP
lbp = local_binary_pattern(image, n_points, radius, method='uniform')

# 显示LBP图像
cv2.imshow('LBP', lbp)
cv2.waitKey(0)
cv2.destroyAllWindows()
