import yolov5 # using a yolo model
import cv2 #for image processing
import numpy as np
from pathlib import Path
import os


# Load the pre-trained football model
model = yolov5.load('keremberke/yolov5m-football')

# Set model parameters
model.conf = 0.25
model.iou = 0.45
model.multi_label = True

input_folder = Path('images/')
output_folder = Path('results/')

image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', 'webp', 'avif']

for extensions in image_extensions:
    for image_path in input_folder.glob(extensions): 
        print(f"Processing {image_path.name}...")

        # Loading the image using OpenCV
        image = cv2.imread(str(image_path))
        image = np.array(image, copy=True) 

        # Object detection
        results = model(image)

        results.render()
        
        save_path = output_folder / image_path.name
        
        # Check if a file with the same name already exists
        if save_path.exists():
            # Add a unique suffix to avoid overwriting (e.g., image1.jpg -> image1_1.jpg)
            base_name = save_path.stem 
            ext = save_path.suffix
            counter = 1
            while (output_folder / f"{base_name}_{counter}{ext}").exists():
                counter += 1
            save_path = output_folder / f"{base_name}_{counter}{ext}"
            
        
        cv2.imwrite(str(save_path), results.ims[0])


        print(results.pandas().xyxy[0])  # Bounding box, class, and confidence data

print("Images saved successfully!")
