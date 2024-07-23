import cv2

from video_processor import VideoProcessor


def main():
    scales = [2, 4, 8]
    video_processor = VideoProcessor(scales)

    video_processor.start()

    while True:
        resized_frames = video_processor.get_resized_frames()

        if resized_frames:
            # Display the resized frames
            for scale, resized_frame in resized_frames:
                window_name = f'Resized Frame (1/{scale})'
                cv2.imshow(window_name, resized_frame)

        # Check if the user wants to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_processor.stop()
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
