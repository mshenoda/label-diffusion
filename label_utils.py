
import random
import cv2
import numpy as np
from PIL import Image

__all__ = ["convert_to_yolo_format", "save_bounding_boxes_yolo_format", 
           "random_unique_int_list", "create_semantic_mask", "create_instance_mask"]

def convert_to_yolo_format(normalized_box):
    x, y, h, w = normalized_box

    # Convert to YOLO format (center_x, center_y, box_width, box_height)
    center_x = x + w / 2
    center_y = y + h / 2
    box_width = w
    box_height = h

    return center_x, center_y, box_width, box_height

def save_bounding_boxes_yolo_format(labels, file_path):
    """
    Save bounding boxes in YOLO format to a text file.

    Parameters:
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        bounding_boxes (list of tuples): List of bounding boxes in (x, y, width, height) format.
        file_path (str): File path to save the YOLO formatted bounding boxes.

    Returns:
        None
    """
    with open(file_path, 'w') as file:
        for label in labels:
            yolo_box = convert_to_yolo_format(label["bounding_box"]["coordinates"])#convert_to_yolo_format(image_width, image_height, box)
            class_id = label["class_id"]
            line = f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n"
            file.write(line)

def save_to_txt_file(file_path, content):
    try:
        with open(file_path, "w") as file:
            file.write(content)
        print(f"String saved to '{file_path}' successfully.")
    except IOError as e:
        print(f"Error: Unable to save the string to '{file_path}'. {e}")

def random_unique_int_list(count, start_range, end_range):
    if count <= 0 or count > (end_range - start_range + 1):
        raise ValueError("Invalid length. It should be a positive integer and less than or equal to the range size.")

    if not isinstance(start_range, int) or not isinstance(end_range, int):
        raise ValueError("Start and end range values should be integers.")

    if start_range >= end_range:
        raise ValueError("Start range should be less than end range.")

    # Generate the list of unique random integers
    random_list = random.sample(range(start_range, end_range + 1), count)

    return random_list

def create_instance_mask(object_mask_coords, image_shape=(512, 512)):
    # Convert object mask coordinates to pixel positions
    pixel_coords = np.array(object_mask_coords).T * np.array(image_shape[::-1]).reshape(2, 1)
    pixel_coords = pixel_coords.astype(int)
    
    # Create an empty binary mask image
    binary_mask = np.zeros(image_shape, dtype=np.uint8)
    
    # Fill the encapsulated regions with white
    cv2.fillPoly(binary_mask, [pixel_coords.T], 255)
    
    return Image.fromarray(binary_mask)

def create_semantic_mask(object_mask_coords_list, image_shape=(512, 512)):
    # Create an empty binary mask image
    binary_mask = np.zeros(image_shape, dtype=np.uint8)
    
    # Fill the encapsulated regions with white for each set of coordinates
    for object_mask_coords in object_mask_coords_list:
        # Convert object mask coordinates to pixel positions
        pixel_coords = np.array(object_mask_coords).T * np.array(image_shape[::-1]).reshape(2, 1)
        pixel_coords = pixel_coords.astype(int)
        
        # Fill the current encapsulated region with white
        cv2.fillPoly(binary_mask, [pixel_coords.T], 255)
    
    return Image.fromarray(binary_mask)