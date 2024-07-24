import time

import cv2

from resize_video_processor import ResizeVideoProcessor


def main():
    scales = [2, 4]
    video_processor = ResizeVideoProcessor(scales)

    video_processor.start()

    # 初始化FPS计算
    fps = 0
    frame_counter = 0
    start_time = time.time()

    while True:
        resized_frames = video_processor.get_resized_frames()

        # 显示分辨率和FPS
        frame_counter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_counter / elapsed_time
            frame_counter = 0
            start_time = time.time()

        if resized_frames:
            # Display the resized frames
            for scale, resized_frame in resized_frames:
                window_name = f'Resized Frame (1/{scale})'
                cv2.putText(resized_frame, f'FPS: {fps:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                            2)
                cv2.imshow(window_name, resized_frame)

        # Check if the user wants to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_processor.stop()
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
