import time

import cv2


def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    # Assume you have 1080p camera, ensure it open with 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # 获取摄像头的分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化两个连续帧
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # 初始化FPS计算
    fps = 0
    frame_counter = 0
    start_time = time.time()

    while cap.isOpened():
        # 对帧进行高斯模糊，模拟周边视觉的低分辨率特性
        blurred_frame1 = cv2.GaussianBlur(frame1, (21, 21), 0)
        blurred_frame2 = cv2.GaussianBlur(frame2, (21, 21), 0)

        # 计算两个连续帧之间的差异
        diff = cv2.absdiff(blurred_frame1, blurred_frame2)

        # 将差异图像转换为灰度图
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # 应用阈值，获得二值化图像
        _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)

        # 膨胀操作去除噪声
        dilated = cv2.dilate(thresh, None, iterations=2)

        # 寻找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 在原始帧上绘制轮廓
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示分辨率和FPS
        frame_counter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_counter / elapsed_time
            frame_counter = 0
            start_time = time.time()

        cv2.putText(frame1, f'Resolution: {width}x{height}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                    2)
        cv2.putText(frame1, f'FPS: {fps:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 显示结果
        cv2.imshow('Motion Detection', frame1)

        # 更新帧
        frame1 = frame2
        ret, frame2 = cap.read()

        # 按q键退出
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
