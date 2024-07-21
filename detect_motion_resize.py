import threading
import time

import cv2


class VideoCaptureAsync:
    def __init__(self, src=0):
        self.thread = None
        self.src = src
        self.cap = cv2.VideoCapture(self.src)

        # Assume you have 1080p camera, ensure it open with 1080p
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.grabbed, self.frame = self.cap.read()
        if not self.grabbed:
            print('[!] Failed to read from camera.')
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())

    def start(self):
        if self.started:
            print('[!] Asynchronous video capturing is already started.')
            return None
        self.started = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return self.grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()
        self.cap.release()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()


def main():
    # 使用异步视频捕获
    cap = VideoCaptureAsync().start()

    # 获取摄像头的分辨率
    width = int(cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, frame1 = cap.read()
    if not ret:
        print('[!] Failed to read from camera.')
        return

    # 缩小图像
    small_frame1 = cv2.resize(frame1, (width // 2, height // 2))

    # 初始化FPS计算
    fps = 0
    frame_counter = 0
    start_time = time.time()

    while True:
        # 读取新帧
        ret, frame2 = cap.read()
        if not ret:
            break

        # 缩小图像
        small_frame2 = cv2.resize(frame2, (width // 2, height // 2))

        # 计算两个连续帧之间的差异
        diff = cv2.absdiff(small_frame1, small_frame2)

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
            if cv2.contourArea(contour) < 1000:  # 调整最小轮廓面积以减少计算量
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x * 2, y * 2), (x * 2 + w * 2, y * 2 + h * 2), (0, 255, 0), 2)

        # 显示分辨率和FPS
        frame_counter += 1
        elapsed_time = time.time() - start_time
        if (elapsed_time > 1):
            fps = frame_counter / elapsed_time
            frame_counter = 0
            start_time = time.time()

        cv2.putText(frame1, f'Resolution: {width}x{height}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                    2)
        cv2.putText(frame1, f'FPS: {fps:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 显示结果
        cv2.imshow('Motion Detection', frame1)

        # 更新缓存的模糊帧和原始帧
        small_frame1 = small_frame2
        frame1 = frame2.copy()

        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
