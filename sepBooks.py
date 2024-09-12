import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def findNumBooks(i):
    # Load the image
    image = cv2.imread('test1.jpg')

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur or try median blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (i, i), 0)
    blurred = cv2.medianBlur(blurred, i)

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply Canny edge detection
    edges = cv2.Canny(thresh, 30, 100)

    # Apply dilation to strengthen the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    im = Image.fromarray(dilated)
    im.show()

    # Find and count rectangular contours (possible books)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_books = 0
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Check if the contour is rectangular
            num_books += 1

    return num_books

findNumBooks(37)




