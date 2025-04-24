import yolov5 # using a yolo model
import cv2 #for image processing
import numpy as np

# Load the pre-trained football model
model = yolov5.load('keremberke/yolov5n-football')

# Set model parameters
model.conf = 0.25
model.iou = 0.45

# Specify the input image
image_path = 'images/pedri.jpg'

# Load the image using OpenCV
image = cv2.imread(image_path)
image = np.array(image, copy=True)  # Ensure the array is writable because when an image is loaded via OpenCV, it becomes a NumPy array, have to ensure is writable for detection.

# Perform object detection
results = model(image)

# Display detected objects
print(results.pandas().xyxy[0])

# Save results to the 'results/' f
results.save(save_dir='results/')

print("Annotated image saved successfully!")
