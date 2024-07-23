import threading

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

    def release(self):
        self.started = False
        self.thread.join()
        self.cap.release()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
