from inference_sdk import InferenceHTTPClient
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Load the image
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
