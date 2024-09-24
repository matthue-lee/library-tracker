import os
import pytesseract
from PIL import Image
import cv2
from inference_sdk import InferenceHTTPClient
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

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
            #image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

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

            # Process without binary inversion for all thresholds
            for threshold in thresholds:
                result_non_inverted, dilated_non_inverted = extract_text(blurred, threshold, False)
                results[filename]['non_inverted'][threshold] = {
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

def display_averarge_confidence(average_confidences):
    for book, orientations in average_confidences.items():
        for orientation, thresholds in orientations.items():
            for threshold, avg_conf in thresholds.items():
                if avg_conf is not None:
                    print(f"Book: {book}, Orientation: {orientation}, Threshold {threshold}: {avg_conf:.2f}")
                else:
                    print(f"Book: {book}, Orientation: {orientation}, Threshold {threshold}: N/A")

def find_overall_best_threshold(average_confidences):
    best_confidences = {}
    
    for book, orientations in average_confidences.items():
        best_threshold = None
        highest_confidence = -1  # Start with a very low confidence
        best_orientation = None
        
        for orientation, thresholds in orientations.items():
            for threshold, avg_conf in thresholds.items():
                if avg_conf is not None and avg_conf > highest_confidence:
                    highest_confidence = avg_conf
                    best_threshold = threshold
                    best_orientation = orientation
        
        best_confidences[book] = {
            'best_orientation': best_orientation,
            'best_threshold': best_threshold,
            'highest_confidence': highest_confidence
        }
    
    return best_confidences

def display_best_text_for_books(results, average_confidences):
    for book, details in average_confidences.items():
        best_orientation = details['best_orientation']
        best_threshold = details['best_threshold']
        
        if best_orientation and best_threshold:
            # Extract the text from the results using the best orientation and threshold
            text = results[book][best_orientation][best_threshold]['text']
            print(f"Book: {book}, Best Orientation: {best_orientation}, Best Threshold: {best_threshold}")
            print(f"Extracted Text: {text}\n")
        else:
            print(f"Book: {book} does not have a valid best threshold.\n")

def identify_books(img_path):
    image = Image.open(img_path)

    # Initialize the inference client
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="XCOXwUNs0horAXScKckM"
    )

    # Perform inference
    result = CLIENT.infer(image, model_id="coco/5")

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)

    # Assuming 'result' contains 'predictions' with 'x', 'y', 'width', 'height'
    for prediction in result['predictions']:
        x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        
        # Calculate the bounding box coordinates
        xmin = x - width / 2
        ymin = y - height / 2
        xmax = x + width / 2
        ymax = y + height / 2
        
        # Draw the rectangle (bounding box)
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=10)

    # Display the image with bounding boxes
    plt.imshow(image)
    plt.axis('off')  # Turn off the axes
    plt.show()
    
    return(result)



def main():
    img_path = 'test2.jpg'
    output_dir = 'output_images'

    #results = identify_books(img_path=img_path)
    #crop_boxes(results, img_path, output_dir)
    
    results = process_images(output_dir, [50,100,150,200])
    average_confidences = calculate_average_confidence(results)
    #display_averarge_confidence(average_confidences)
    best_guess = find_overall_best_threshold(average_confidences)
    display_best_text_for_books(results, best_guess)



if __name__ == "__main__":
    main()