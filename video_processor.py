import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue

import cv2

from resizer import FrameResizer


class VideoProcessor:
    def __init__(self, scales):
        self.future = None
        self.executor = None
        self.queue = Queue()
        self.stop_event = threading.Event()
        self.frame_resizer = FrameResizer(scales)

    def capture_and_process_video(self):
        # Open the camera (default camera index is 0)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        with ProcessPoolExecutor() as executor:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                # Submit tasks to resize the frame
                futures = [executor.submit(self.frame_resizer.resize_frame, frame, scale) for scale in
                           self.frame_resizer.scales]

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
