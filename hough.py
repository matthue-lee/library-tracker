import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load Image
def load_image(image_path):
    return cv2.imread(image_path)

# Preprocess Image: Grayscale and Edge Detection
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    return edges

# Perform Hough Line Transform to Detect Lines
def detect_lines(edges):
    # Use HoughLinesP for probabilistic line transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=150, minLineLength=200, maxLineGap=20)
    return lines

# Draw Lines on Original Image
def draw_lines(image, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

# Main Processing
def main(image_path):
    # Step 1: Load the image
    image = load_image(image_path)

    # Step 2: Preprocess the image (grayscale and edge detection)
    edges = preprocess_image(image)

    # Step 3: Detect lines using Hough Transform
    lines = detect_lines(edges)

    # Step 4: Draw the detected lines on the original image
    image_with_lines = draw_lines(image.copy(), lines)

    # Step 5: Display the image with detected lines
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for plotting
    plt.title("Hough Line Transform for Book Detection")
    plt.show()

# Example Usage
image_path = 'test1.jpg'  # Your image file path
main(image_path)
