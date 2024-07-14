import cv2
import numpy as np

# 读取图像
image = cv2.imread('images/image_0107.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Helper function to find top 3 areas
def find_top_areas(mask, image):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [(cv2.contourArea(c), c) for c in contours]
    areas = sorted(areas, key=lambda x: x[0], reverse=True)[:3]
    top_areas = [cv2.boundingRect(a[1]) for a in areas]
    return [image[y:y+h, x:x+w] for x, y, w, h in top_areas]

# 提取颜色的top 3子图像
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # 红、绿、蓝
color_masks = [cv2.inRange(image, np.array(color)-50, np.array(color)+50) for color in colors]
color_top_areas = [find_top_areas(mask, image) for mask in color_masks]

# 提取对比度的top 3子图像
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
contrast = np.uint8(np.absolute(laplacian))
_, contrast_mask = cv2.threshold(contrast, 128, 255, cv2.THRESH_BINARY)
contrast_top_areas = find_top_areas(contrast_mask, image)

# 提取饱和度的top 3子图像
saturation = hsv[:, :, 1]
_, saturation_mask = cv2.threshold(saturation, 128, 255, cv2.THRESH_BINARY)
saturation_top_areas = find_top_areas(saturation_mask, image)

# 提取亮度的top 3子图像
brightness = hsv[:, :, 2]
_, brightness_mask = cv2.threshold(brightness, 128, 255, cv2.THRESH_BINARY)
brightness_top_areas = find_top_areas(brightness_mask, image)

# 保存结果
for i, areas in enumerate(color_top_areas):
    for j, area in enumerate(areas):
        cv2.imwrite(f'output/color_{i+1}_top_{j+1}.jpg', area)

for i, area in enumerate(contrast_top_areas):
    cv2.imwrite(f'output/contrast_top_{i+1}.jpg', area)

for i, area in enumerate(saturation_top_areas):
    cv2.imwrite(f'output/saturation_top_{i+1}.jpg', area)

for i, area in enumerate(brightness_top_areas):
    cv2.imwrite(f'output/brightness_top_{i+1}.jpg', area)
