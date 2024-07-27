import cv2
import numpy as np


def find_dominant_colors(image, num_colors=5, threshold=10):
    # 将图像从BGR转换为HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 计算色相通道的直方图
    hist = cv2.calcHist([image_hsv], [0], None, [180], [0, 180])

    # 找到直方图的峰值
    hist = cv2.normalize(hist, hist).flatten()
    peaks = np.argsort(hist)[-num_colors:]

    # 获取主要颜色的色相值
    dominant_colors = []
    for peak in peaks:
        h = peak
        dominant_colors.append((h, 255, 255))  # 将饱和度和亮度设置为最大值

    # 合并相似颜色
    merged_colors = []
    for color in dominant_colors:
        if not merged_colors:
            merged_colors.append(color)
        else:
            is_similar = False
            for m_color in merged_colors:
                if abs(color[0] - m_color[0]) < threshold:
                    is_similar = True
                    break
            if not is_similar:
                merged_colors.append(color)

    return merged_colors


def create_color_mask(image, color, tolerance=10):
    # 定义颜色范围（考虑环绕的情况）
    lower1 = np.array([max(color[0] - tolerance, 0), 50, 50])
    upper1 = np.array([color[0], 255, 255])
    lower2 = np.array([color[0], 50, 50])
    upper2 = np.array([min(color[0] + tolerance, 179), 255, 255])

    # 创建掩码
    mask1 = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), lower1, upper1)
    mask2 = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), lower2, upper2)

    return cv2.bitwise_or(mask1, mask2)


# 加载图像
image = cv2.imread('image.jpg')

# 找到主要颜色
num_colors = 5
dominant_colors = find_dominant_colors(image, num_colors)

# 提取每个主要颜色的色块
for i, color in enumerate(dominant_colors):
    mask = create_color_mask(image, color)
    color_block = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow(f'Color Block {i + 1}', color_block)
    cv2.imwrite(f'color_block_{i + 1}.png', color_block)

cv2.waitKey(0)
cv2.destroyAllWindows()
