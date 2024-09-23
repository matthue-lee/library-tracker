from roboflow import *
import os
import pytesseract
from PIL import Image
import cv2

def crop_boxes(results, img_path, output_folder):
    # Open the original image
    image = Image.open(img_path)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each prediction (bounding box)
    for i, prediction in enumerate(results['predictions']):
        x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']

        # Calculate the bounding box coordinates
        xmin = int(x - width / 2)
        ymin = int(y - height / 2)
        xmax = int(x + width / 2)
        ymax = int(y + height / 2)

        # Crop the image to the bounding box
        cropped_image = image.crop((xmin, ymin, xmax, ymax))

        # Save the cropped image in the specified folder
        cropped_image.save(os.path.join(output_folder, f"book_{i+1}.jpg"))  # Save each crop as a new image

        # Optionally display the cropped image
        # cropped_image.show()


def process_images(input_folder, thresholds):
    results = {}
    
    # Iterate over each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Convert to grayscale and apply Gaussian Blur
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Process each threshold for both inversion and non-inversion
            results[filename] = {'inverted': {}, 'non_inverted': {}}
            
            for threshold in thresholds:
                # Process with binary inversion
                result, dilated_image = extract_text(blurred, threshold, True)
                results[filename]['inverted'][threshold] = {
                    'text': result['text'],
                    'confidence': result['confidence'],
                }

            # Process without binary inversion
            result_non_inverted, dilated_non_inverted = extract_text(blurred, 100, False)
            results[filename]['non_inverted'] = {
                'text': result_non_inverted['text'],
                'confidence': result_non_inverted['confidence'],
            }

    return results

# Helper function to extract text from the image with confidence
def extract_text(blurred, threshold_value, use_inversion):
    if use_inversion:
        _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    # Dilate to improve clarity of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Run Pytesseract with custom config
    custom_config = r'--psm 6'
    details = pytesseract.image_to_data(dilated, config=custom_config, output_type=pytesseract.Output.DICT)

    # Extract text and confidence values
    result = {'text': '', 'confidence': []}
    for i in range(len(details['text'])):
        if int(details['conf'][i]) > 0:  # Filter low-confidence detections
            result['text'] += details['text'][i] + ' '
            result['confidence'].append(details['conf'][i])

    return result, dilated

def calculate_average_confidence(book_data):
    averages = {}
    for book, data in book_data.items():
        averages[book] = {}
        for orientation in ['inverted', 'non_inverted']:
            if orientation in data:
                averages[book][orientation] = {}
                for threshold, content in data[orientation].items():
                    if 'confidence' in content and content['confidence']:
                        avg_conf = sum(content['confidence']) / len(content['confidence'])
                        averages[book][orientation][threshold] = avg_conf
                    else:
                        averages[book][orientation][threshold] = None  # or some default value
    return averages


def main():
    img_path = 'test1.jpg'
    output_dir = 'output_images'
    #results = identify_books(img_path=img_path)
    #crop_boxes(results, img_path, output_dir)
    results = process_images(output_dir, [50,100,150,200])
    print(results)
    average_confidences = calculate_average_confidence(results)

    # Output the average confidences for each book and orientation
    for book, orientations in average_confidences.items():
        for orientation, thresholds in orientations.items():
            for threshold, avg_conf in thresholds.items():
                if avg_conf is not None:
                    print(f"Book: {book}, Orientation: {orientation}, Threshold {threshold}: {avg_conf:.2f}")
                else:
                    print(f"Book: {book}, Orientation: {orientation}, Threshold {threshold}: N/A")


if __name__ == "__main__":
    main()