import cv2
import numpy as np

def quantize(value, bins):
    """Quantize the value into one of the bins."""
    bin_size = 256 // bins
    return value // bin_size


def extract_top_segments(image_path, num_bins=32, top_n=3, sv_threshold=64):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply the threshold to the S and V channels
    s_mask = hsv_image[:, :, 1] > sv_threshold
    v_mask = hsv_image[:, :, 2] > sv_threshold
    valid_mask = s_mask & v_mask

    # Quantize the HSV values into bins
    h_quantized = quantize(hsv_image[:, :, 0], num_bins)
    s_quantized = quantize(hsv_image[:, :, 1], num_bins)
    v_quantized = quantize(hsv_image[:, :, 2], num_bins)

    # Function to find top N segments in a quantized channel
    def find_top_n_segments(quantized_channel, valid_mask):
        quantized_channel = quantized_channel[valid_mask]
        unique, counts = np.unique(quantized_channel, return_counts=True)
        sorted_indices = np.argsort(-counts)
        return unique[sorted_indices[:top_n]]

    # Find top N segments for each channel
    top_h_segments = find_top_n_segments(h_quantized, valid_mask)
    top_s_segments = find_top_n_segments(s_quantized, valid_mask)
    top_v_segments = find_top_n_segments(v_quantized, valid_mask)

    # Create masks for the top segments and apply them to the original image
    def apply_masks(quantized_channel, top_segments, valid_mask):
        masks = [((quantized_channel == seg) & valid_mask) for seg in top_segments]
        return [cv2.bitwise_and(image, image, mask=mask.astype(np.uint8) * 255) for mask in masks]

    top_h_images = apply_masks(h_quantized, top_h_segments, valid_mask)
    top_s_images = apply_masks(s_quantized, top_s_segments, valid_mask)
    top_v_images = apply_masks(v_quantized, top_v_segments, valid_mask)

    return top_h_images, top_s_images, top_v_images

# Example usage
image_path = 'images/color_picker.png'  # Replace with your image path
top_h_images, top_s_images, top_v_images = extract_top_segments(image_path)

# Save or display the resulting images
for i, img in enumerate(top_h_images):
    cv2.imwrite(f'output/top_h_segment_{i + 1}.png', img)

for i, img in enumerate(top_s_images):
    cv2.imwrite(f'output/top_s_segment_{i + 1}.png', img)

for i, img in enumerate(top_v_images):
    cv2.imwrite(f'output/top_v_segment_{i + 1}.png', img)

# Uncomment the following lines to display the images
# for i, img in enumerate(top_h_images):
#     cv2.imshow(f'Top H Segment {i + 1}', img)
# for i, img in enumerate(top_s_images):
#     cv2.imshow(f'Top S Segment {i + 1}', img)
# for i, img in enumerate(top_v_images):
#     cv2.imshow(f'Top V Segment {i + 1}', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
