from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2


def resize_frame(frame, scale):
    """
    Resize the frame to the given scale.
    """
    height, width = frame.shape[:2]
    new_dimensions = (int(width / scale), int(height / scale))
    resized_frame = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_AREA)
    return scale, resized_frame


def capture_and_process_video(scales):
    # Open the camera (default camera index is 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    with ProcessPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Submit tasks to resize the frame
            futures = [executor.submit(resize_frame, frame, scale) for scale in scales]

            # Collect the results
            resized_frames = []
            for future in as_completed(futures):
                resized_frames.append(future.result())

            # Display the resized frames
            for scale, resized_frame in resized_frames:
                window_name = f'Resized Frame (1/{scale})'
                cv2.imshow(window_name, resized_frame)

            # Check if the user wants to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    scales = [2, 4, 8]
    capture_and_process_video(scales)
