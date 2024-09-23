import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load Image
def load_image(image_path):
    return cv2.imread(image_path)

# Preprocess Image: Grayscale and Edge Detection
def preprocess_image(image, blur_value, canny_threshold1, canny_threshold2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)  # Apply Gaussian blur
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)  # Canny edge detection
    return edges

# Save image
def save_image(image, iteration, output_dir='output_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, f"processed_image_{iteration}.jpg")
    cv2.imwrite(save_path, image)
    print(f"Saved: {save_path}")

# Filter lines by length
def filter_lines_by_length(lines, min_length=200):
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if line_length > min_length:  # Only keep lines longer than the threshold
                filtered_lines.append(line)
    return filtered_lines

# Draw Lines on Original Image
def draw_lines(image, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

# Iterate over different parameter settings
def iterate_parameters(image_path):
    # Load the image
    image = load_image(image_path)

    # Iterate over different blur values, Canny thresholds, and Hough Transform parameters
    for blur_value in range(7, 17, 2):  # Iterate over odd blur values from 7 to 15
        for canny_threshold1 in range(30, 151, 30):  # Lower range for Canny threshold1
            for canny_threshold2 in range(60, 301, 60):  # Lower range for Canny threshold2
                for min_line_length in [100, 200, 500]:  # Test different minLineLength values
                    for max_line_gap in [100, 200, 300]:  # Test different maxLineGap values
                        # Preprocess the image (grayscale, blur, and edge detection)
                        edges = preprocess_image(image, blur_value, canny_threshold1, canny_threshold2)

                        # Use Hough Transform to detect lines
                        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80,
                                                minLineLength=min_line_length, maxLineGap=max_line_gap)

                        # Filter lines by length (optional, can be adjusted or omitted)
                        filtered_lines = filter_lines_by_length(lines, min_length=min_line_length)

                        # Draw the filtered lines on the original image
                        image_with_lines = draw_lines(image.copy(), filtered_lines)

                        # Save the image with different parameter settings
                        save_image(image_with_lines, f"blur_{blur_value}_canny_{canny_threshold1}_{canny_threshold2}_minlen_{min_line_length}_maxgap_{max_line_gap}")

                        # Optionally display the processed image
                        # plt.figure(figsize=(8, 8))
                        # plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
                        # plt.title(f"Blur: {blur_value}, Canny: {canny_threshold1}/{canny_threshold2}, MinLen: {min_line_length}, MaxGap: {max_line_gap}")
                        # plt.show()

# Example Usage
image_path = 'test1.jpg'  # Your image file path
iterate_parameters(image_path)
