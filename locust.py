import cv2
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt

# Load and rotate the image
book_dir = 'output_images/book_4.jpg'
image = cv2.imread(book_dir)
image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Function to process image and extract text with confidence
def process_image(blurred, threshold_value, use_inversion):
    if use_inversion:
        _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    # Optionally, you can dilate/erode to improve clarity of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Run Pytesseract with custom config (using PSM 6)
    custom_config = r'--psm 6'
    details = pytesseract.image_to_data(dilated, config=custom_config, output_type=pytesseract.Output.DICT)

    # Extract text and confidence values
    result = {
        'text': '',
        'confidence': [],
    }

    for i in range(len(details['text'])):
        if int(details['conf'][i]) > 0:  # Filter out low-confidence detections
            result['text'] += details['text'][i] + ' '
            result['confidence'].append(details['conf'][i])

    return result, dilated

# Threshold values for inversion
threshold_values = [50, 100, 150, 200]

# Process the image with binary inversion for each threshold value
results_inverted = {}
for threshold in threshold_values:
    result, dilated_image = process_image(blurred, threshold, True)
    results_inverted[threshold] = {
        'text': result['text'],
        'confidence': result['confidence'],
        'dilated_image': dilated_image
    }

# Process the image without binary inversion using a single threshold
result_non_inverted, dilated_non_inverted = process_image(blurred, 100, False)

# Output results for inverted thresholds
with open('extracted_text.txt', 'w') as f:
    for threshold, result in results_inverted.items():
        f.write(f'--- With Binary Inversion (Threshold: {threshold}) ---\n')
        f.write(result['text'] + '\n')
        f.write('Confidence Scores: ' + ', '.join(map(str, result['confidence'])) + '\n\n')

    # Non-inverted results
    f.write('--- Without Binary Inversion ---\n')
    f.write(result_non_inverted['text'] + '\n')
    f.write('Confidence Scores: ' + ', '.join(map(str, result_non_inverted['confidence'])) + '\n')

# Save the preprocessed images for each threshold value
for threshold, result in results_inverted.items():
    cv2.imwrite(f'preprocessed_inverted_{threshold}.png', result['dilated_image'])

# Save the non-inverted image
cv2.imwrite('preprocessed_non_inverted.png', dilated_non_inverted)