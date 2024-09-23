
import cv2
import numpy as np

def find_rectangles(image):
        
    # Visualize original image
    # cv2.imshow("Original Image", cv2.resize(image, None, fx=0.5, fy=0.5))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Convert the image to grayscale
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # This gets rid of the background gray lines.
    _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_TOZERO_INV)

    # # Visualize thresholded image
    # cv2.imshow("Thresholded Image", cv2.resize(binary, None, fx=0.5, fy=0.5))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # Find edges using Canny edge detection
    edges = cv2.Canny(binary, 50, 150)

    # Visualize edges
    cv2.imshow("Edges", cv2.resize(edges, (750,1000), fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

    # Find lines using probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=1, maxLineGap=15)

    # Create an empty mask to draw lines
    mask = np.zeros_like(gray)

    # Draw lines on the mask
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(mask, (x1, y1), (x2, y2), 255, 2)

    # Visualize Hough lines
    cv2.imshow("Hough Lines", cv2.resize(mask, (750,1000), fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply morphological operations to enhance rectangles
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=6)
    eroded_mask = cv2.erode(dilated_mask, kernel, iterations=5)

    # Visualize enhanced mask
    cv2.imshow("Enhanced Mask", cv2.resize(eroded_mask, (750,1000), fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find contours in the mask
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    # Filter contours by shape
    rectangles = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(approx)
            rectangles.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "color": cv2.mean(img[y:y + h, x:x + w])
            })

    return rectangles


rectangles = find_rectangles('test1.jpg')
cv2.imshow(rectangles)
cv2.waitKey(0)
cv2.destroyAllWindows()