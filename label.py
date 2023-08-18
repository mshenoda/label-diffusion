#   LabelDiffusion - Automatic Labeling of Stable Diffusion Pipelines
#   Copyright (C) 2023  Michael Shenoda
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published
#   by the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from .label_utils import create_semantic_mask
from .image_processing import threshold

__all__ = ["label_attention_map", "label_image"]

def calculate_score(image:np.ndarray, bbox:tuple, intensity_weight=0.5, size_weight=0.5, max_object_size=0.99):
    """
    Calculate a more robust bounding box score based on heatmap 

    Parameters:
    image (numpy.ndarray): The input heatmap image in OpenCV format (grayscale).
    bbox (tuple): Normalized bounding box coordinates (x, y, w, h).
    intensity_weight (float): Weight for the intensity features in the final score.
    size_weight (float): Weight for the object size compared to the image size in the final score.
    max_object_size (float): The maximum expected object size as a fraction of the image size.

    Returns:
    float: The bounding box score, constrained between 0 and 1.
    """
    # Convert normalized bounding box to pixel coordinates
    height, width = image.shape[:2]
    x, y, w, h = bbox
    x1, y1 = int(x * width), int(y * height)
    x2, y2 = int((x + w) * width), int((y + h) * height)

    # Crop the bounding box area from the image
    cropped_area = image[y1:y2, x1:x2]

    # Calculate the average intensity of the cropped area
    average_intensity = np.mean(cropped_area)

    # Calculate the maximum intensity within the bounding box
    max_intensity = np.max(cropped_area)

    # Calculate the histogram of pixel intensities within the bounding box
    hist, _ = np.histogram(cropped_area, bins=256, range=(0, 255))

    # Normalize the average intensity, maximum intensity, number of hot (high-intensity) pixels, and object size
    average_intensity_score = average_intensity / 255.0
    max_intensity_score = max_intensity / 255.0
    hot_histogram_bins_score = np.sum(hist[127:]) / (cropped_area.size)
    object_size_score = min(w * h / (width * height), max_object_size)

    # Calculate the total weighted sum of the features
    total_weighted_sum = (
        intensity_weight * average_intensity_score +
        intensity_weight * max_intensity_score +
        intensity_weight * hot_histogram_bins_score +
        size_weight * object_size_score
    )

    # Normalize the total weighted sum to get the final bounding box score (between 0 and 1)
    bounding_box_score = total_weighted_sum / (intensity_weight + size_weight)
    
    # Ensure the score is within the range of 0 to 1
    bounding_box_score = np.clip(bounding_box_score, 0.0, 1.0)

    return bounding_box_score

def compute_labels(binary_mask, attention_map, min_blob_area, max_blob_area):
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate average blob area for filtering outliers
    total_area = sum(cv2.contourArea(contour) for contour in contours)
    average_area = total_area / len(contours)

    # Get the height and width of the original attention_map
    height, width = attention_map.shape[:2]

    # Create list of labels with normalized bounding boxes, bounding polygons, and coordinate lists
    labels = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_blob_area <= area <= (max_blob_area or float('inf')) and area >= 0.05 * average_area:
            x, y, w, h = cv2.boundingRect(contour)
            x_n = x / width
            y_n = y / height
            w_n = w / width
            h_n = h / height
            cx_n = (x + w / 2) / width
            cy_n = (y + h / 2) / height

            # Get the bounding polygon in normalized coordinates
            contour_normalized = contour.squeeze() / [width, height]
            bounding_polygon_normalized = contour_normalized.tolist()
            bounding_box_coordinates = [x_n, y_n, w_n, h_n]
            label = {
                'score': calculate_score(attention_map, bounding_box_coordinates),
                "class_id": 0,
                'bounding_box': {
                    'x': x_n,
                    'y': y_n,
                    'cx': cx_n,
                    'cy': cy_n,
                    'width': w_n,
                    'height': h_n,
                    'coordinates': bounding_box_coordinates
                },
                'bounding_polygon': bounding_polygon_normalized
            }
            labels.append(label)
    return labels

def label_attention_map(attention_map, threshold_value=127, min_blob_area=50, max_blob_area=None):
    semantic_mask, attention_map = threshold(attention_map, threshold_value)
    labels = compute_labels(semantic_mask, attention_map, min_blob_area, max_blob_area)
    return labels, Image.fromarray(semantic_mask)

def label_image(image:Image, yolo:YOLO, class_id_mapping, confidence=0.45, image_size=640, rect_image=True, cuda_device=0):
    results = yolo.predict(source=image, conf=confidence, show=False, save_txt=False, save=False, save_crop=False, imgsz=image_size, rect=rect_image, device=cuda_device)
    
    class_mapping = None
    if isinstance(class_id_mapping, dict):
        class_mapping = class_id_mapping
    elif isinstance(class_id_mapping, list):
        class_mapping = {i: i for i in class_id_mapping}

    # Process results list
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.xywhn.tolist()
            classes = result.boxes.cls.tolist()
            classes = [int(x) for x in classes]
            confidences = result.boxes.conf.tolist()
        else:
            return list(), Image.new("L", image.size, 0)

        if hasattr(result, 'masks') and result.masks is not None and hasattr(result.masks, 'xyn'):
            masks = result.masks.xyn
        else:
            return list(), Image.new("L", image.size, 0)
        
        target_masks = []
        labels = []
        if class_mapping is None:
            class_mapping = {i: i for i in classes}
        for i in range(len(boxes)):
            if classes[i] in class_mapping.values():
                target_masks.append(masks[i])
                cx_n, cy_n, w_n, h_n = boxes[i]
                x_n = cx_n - w_n / 2
                y_n = cy_n - h_n / 2
                label = {
                    "score": confidences[i],
                    "class_id": classes[i],
                    "bounding_box": {
                        "x": x_n,
                        "y": y_n,
                        "cx": cx_n,
                        "cy": cy_n,
                        "width": w_n,
                        "height": h_n,
                        "coordinates": [x_n, y_n, w_n, h_n]
                    },
                    "bounding_polygon": masks[i].tolist()
                }
                labels.append(label)
        semantic_mask = create_semantic_mask(target_masks, image.size)

    return labels, semantic_mask
