import cv2


class FrameResizer:
    def __init__(self, scales):
        self.scales = scales

    @staticmethod
    def resize_frame(frame, scale):
        """
        Resize the frame to the given scale.
        """
        height, width = frame.shape[:2]
        new_dimensions = (int(width / scale), int(height / scale))
        resized_frame = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_AREA)
        return scale, resized_frame

    def resize_frames(self, frame):
        resized_frames = []
        for scale in self.scales:
            resized_frames.append(self.resize_frame(frame, scale))
        return resized_frames
