import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import os
import cv2  # OpenCV for image processing

# Load Faster R-CNN Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and Preprocess Image using OpenCV
def preprocess_image_opencv(image_path, blur_ksize=5):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)
    
    # Convert back to 3-channel RGB for the model
    rgb_image = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2RGB)
    
    # Convert to PIL image and then tensor
    pil_image = Image.fromarray(rgb_image)
    image_tensor = F.to_tensor(pil_image).unsqueeze(0)
    
    return image_tensor

# Perform Inference
def get_predictions(image_tensor):
    with torch.no_grad():
        predictions = model(image_tensor)
    print(predictions)  # Inspect predictions
    return predictions

# Apply Non-Maximum Suppression (NMS)
def apply_nms(boxes, scores, iou_threshold=0.3):
    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(scores)
    indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)
    return boxes_tensor[indices].tolist()

# Extract Boxes and Apply NMS
def extract_boxes(predictions, score_threshold=0.3, nms_iou_threshold=0.3):
    boxes = predictions[0]['boxes'].tolist()
    scores = predictions[0]['scores'].tolist()
    
    # Filter by score threshold
    filtered_boxes = [box for i, box in enumerate(boxes) if scores[i] > score_threshold]
    filtered_scores = [score for score in scores if score > score_threshold]
    
    # Apply Non-Maximum Suppression
    nms_boxes = apply_nms(filtered_boxes, filtered_scores, nms_iou_threshold)
    
    return nms_boxes

# Define Aspect Ratio Filtering
def filter_boxes_by_aspect_ratio(boxes, min_aspect_ratio=0.3, max_aspect_ratio=3.0):  # Broader range
    filtered_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        if height > 0:
            aspect_ratio = width / height
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                filtered_boxes.append(box)
    return filtered_boxes

# Draw Bounding Boxes
def draw_boxes(image_path, boxes):
    print("Identified:", len(boxes), "books.")
    image = Image.open(image_path)
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    ax = plt.gca()
    for box in boxes:
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                             fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(rect)
    plt.show()

# Save Cropped Boxes
def save_cropped_boxes(image_path, boxes, output_dir='cropped_boxes'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image = Image.open(image_path)
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        cropped_image_path = os.path.join(output_dir, f'box_{i}.jpg')
        cropped_image.save(cropped_image_path)
        print(f'Saved cropped box {i} to {cropped_image_path}')

# Main Processing
image_path = 'test1.jpg'  # Path to your uploaded image
image_tensor = preprocess_image_opencv(image_path, blur_ksize=5)  # Preprocess using OpenCV
predictions = get_predictions(image_tensor)
boxes = extract_boxes(predictions, score_threshold=0.5)  # Increase score threshold for stricter detection
filtered_boxes = filter_boxes_by_aspect_ratio(boxes)
draw_boxes(image_path, filtered_boxes)
save_cropped_boxes(image_path, filtered_boxes)  # Save cropped boxes
