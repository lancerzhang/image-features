import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue

import cv2


def process_frame(frame, scale, motion_scale):
    """
    Resize the frame to the given scale.
    """
    height, width = frame.shape[:2]
    new_dimensions = (int(width / scale), int(height / scale))
    resized_frame = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_AREA)
    contours = None
    # TODO, how to save and read prev_frame?
    # if scale == motion_scale
    #   contours = detect_motion(prev_frame, resized_frame), how to store and get prev_frame?
    return scale, resized_frame, contours


def detect_motion(prev_frame, current_frame):
    # 计算两个连续帧之间的差异
    diff = cv2.absdiff(prev_frame, current_frame)

    # 将差异图像转换为灰度图
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # 应用阈值，获得二值化图像
    _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)

    # 膨胀操作去除噪声
    dilated = cv2.dilate(thresh, None, iterations=2)

    # 寻找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


class VideoProcessor:
    def __init__(self, scales):
        self.scales = scales
        self.future = None
        self.executor = None
        self.queue = Queue()
        self.stop_event = threading.Event()

    def capture_and_process_video(self):
        # Open the camera (default camera index is 0)
        cap = cv2.VideoCapture(0)

        # Assume you have 1080p camera, ensure it open with 1080p
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        with ProcessPoolExecutor() as executor:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                # Submit tasks to resize the frame
                # TODO, get motion_scale??
                futures = [executor.submit(process_frame, frame, scale, motion_scale) for scale in self.scales]

                # Collect the results and put them in the queue
                resized_frames = []
                for future in as_completed(futures):
                    resized_frames.append(future.result())

                self.queue.put(resized_frames)

        # Release the camera
        cap.release()

    def start(self):
        # Start the capture_and_process_video in a separate thread
        self.executor = ThreadPoolExecutor()
        self.future = self.executor.submit(self.capture_and_process_video)

    def stop(self):
        # Stop the video capture and processing
        self.stop_event.set()
        self.future.result()  # Ensure the capture_and_process_video thread has finished
        self.executor.shutdown()

    def get_resized_frames(self):
        if not self.queue.empty():
            return self.queue.get()
        return None

    def set_motion_scale(self, scale):
        # TODO, set motion_scale for process_frame method
        pass
