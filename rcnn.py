import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and preprocess the image
image_path = "test1.jpg"  # Replace with your image path
image = Image.open(image_path)
image_tensor = F.to_tensor(image).unsqueeze(0)  # Add a batch dimension

# Make predictions
with torch.no_grad():
    predictions = model(image_tensor)

# Filter predictions
boxes = predictions[0]['boxes']  # Bounding boxes
scores = predictions[0]['scores']  # Confidence scores
threshold = 0.3  # Set a threshold for filtering boxes
filtered_boxes = boxes[scores > threshold]

# Visualize the results
fig, ax = plt.subplots(1)
ax.imshow(image)

for box in filtered_boxes:
    xmin, ymin, xmax, ymax = box
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.axis('off')
plt.show()
