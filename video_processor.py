import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from queue import Queue

import cv2


def detect_motion(prev_frame, current_frame):
    # Compute the difference between the two frames
    diff = cv2.absdiff(prev_frame, current_frame)

    # Convert the difference image to grayscale
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get a binary image
    _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)

    # Dilate the image to remove noise
    dilated = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def process_frame(frame, scale, motion_scale, prev_frame_container):
    """
    Resize the frame to the given scale.
    """
    height, width = frame.shape[:2]
    new_dimensions = (int(width / scale), int(height / scale))
    resized_frame = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_AREA)
    contours = None

    # Detect motion if the scale matches the motion_scale
    if scale == motion_scale:
        if prev_frame_container['prev_frame'] is not None:
            contours = detect_motion(prev_frame_container['prev_frame'], resized_frame)
        prev_frame_container['prev_frame'] = resized_frame

    return scale, resized_frame, contours


class VideoProcessor:
    def __init__(self, scales):
        self.scales = scales
        self.future = None
        self.thread_executor = None
        self.queue = Queue()
        self.is_running = True
        self.motion_scale = None
        self.manager = Manager()
        self.prev_frame_container = self.manager.dict({'prev_frame': None})

    def capture_and_process_video(self):
        # Open the camera (default camera index is 0)
        cap = cv2.VideoCapture(0)

        # Assume you have a 1080p camera, ensure it opens with 1080p resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        with ProcessPoolExecutor(max_workers=len(self.scales)) as process_executor:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Submit tasks to resize the frame
                futures = [process_executor.submit(process_frame, frame, scale, self.motion_scale,
                                                   self.prev_frame_container) for scale in self.scales]

                # Collect the results and put them in the queue
                resized_frames = []
                for future in as_completed(futures):
                    resized_frames.append(future.result())

                resized_frames.append((1, frame, None))
                self.queue.put(resized_frames)

                time.sleep(0.01)  # 添加睡眠时间以减少CPU占用

        # Release the camera
        cap.release()

    def start(self):
        # Start the capture_and_process_video in a separate thread
        self.thread_executor = ThreadPoolExecutor()
        self.future = self.thread_executor.submit(self.capture_and_process_video)

    def stop(self):
        # Stop the video capture and processing
        self.is_running = False
        self.future.result()  # Ensure the capture_and_process_video thread has finished
        self.thread_executor.shutdown()

    def get_resized_frames(self):
        if not self.queue.empty():
            return self.queue.get()
        return None

    def set_motion_scale(self, scale):
        self.motion_scale = scale


# Example of detect_motion function
def detect_motion(prev_frame, current_frame):
    # Implement motion detection logic here
    # For example, using frame differencing, thresholding, and contour detection
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(gray_prev, gray_current)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
