import time

import cv2

from video_processor import VideoProcessor

num_scales = 3  # 3 = [2,4,8] scales
motion_scale = 2


def main():
    video_processor = VideoProcessor(num_scales)

    video_processor.start()
    video_processor.set_motion_scale(motion_scale)

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
            raw_frame = None
            motion_contours = None

            # Display the resized frames
            for scale, resized_frame, contours in resized_frames:
                if scale == 1:
                    raw_frame = resized_frame
                    continue
                if scale == motion_scale:
                    motion_contours = contours

                window_name = f'Resized Frame (1/{scale})'
                cv2.imshow(window_name, resized_frame)

            if raw_frame is None or motion_contours is None:
                continue

            # 在原始帧上绘制轮廓
            for contour in motion_contours:
                if cv2.contourArea(contour) < 1000:  # 调整最小轮廓面积以减少计算量
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(raw_frame, (x * motion_scale, y * motion_scale),
                              (x * motion_scale + w * motion_scale, y * motion_scale + h * motion_scale),
                              (0, 255, 0), 2)

            cv2.putText(raw_frame, f'FPS: {fps:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 显示结果
            cv2.imshow('Motion Detection', raw_frame)

        # Check if the user wants to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_processor.stop()
            break
        time.sleep(0.01)  # 添加睡眠时间以减少CPU占用

    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
