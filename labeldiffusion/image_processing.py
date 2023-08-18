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
from scipy.ndimage import label
from PIL import Image

__all__ = ["apply_mask", "concat", "threshold"]

def threshold(attention_map, threshold_value):
    # Convert the attention_map to NumPy array if it's a Pillow Image
    if isinstance(attention_map, Image.Image):
        attention_map = np.array(attention_map)

    # Convert the attention_map to grayscale if it's in color
    if len(attention_map.shape) == 3:
        attention_map = cv2.cvtColor(attention_map, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, binary_mask = cv2.threshold(attention_map, threshold_value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Erosion to remove small noisy regions
    kernel_erode = np.ones((3, 3), np.uint8)  # You can adjust the kernel size as needed
    thresholded_image_eroded = cv2.erode(binary_mask, kernel_erode, iterations=10)

    # Further refine by dilation
    kernel_dilate = np.ones((3, 3), np.uint8)  # You can adjust the kernel size as needed
    binary_mask_final = cv2.dilate(thresholded_image_eroded, kernel_dilate, iterations=7)

    # Fill holes using morphological closing operation
    # kernel_close = np.ones((5, 5), np.uint8)  # You can adjust the kernel size as needed
    # binary_mask_final = cv2.morphologyEx(binary_mask_final, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    return binary_mask_final, attention_map


def filter_tiny_blobs(binary_mask, min_blob_size=512):
    """
    Filter out tiny blobs in a binary mask.

    Parameters:
        binary_mask (ndarray): A 2D binary mask containing blobs.
        min_blob_size (int): The minimum size of blobs to keep (in pixels).

    Returns:
        ndarray: The filtered binary mask with tiny blobs removed.
    """
    # Find connected components in the binary mask
    labeled_mask, num_features = label(binary_mask)

    # Calculate the sizes of the connected components
    component_sizes = [(labeled_mask == i).sum() for i in range(1, num_features + 1)]

    # Create a mask to keep only the components with sizes above the threshold
    mask_to_keep = np.isin(labeled_mask, np.nonzero(np.array(component_sizes) >= min_blob_size)[0] + 1)

    # Extract the regions of the filtered blobs
    filtered_mask = binary_mask.copy()
    filtered_mask[~mask_to_keep] = 0

    return filtered_mask

def enhance(img, threshold, gamma=1.25):
    img = cv2.pow(img / 255.0, gamma) * 255.0

    img = cv2.addWeighted(img, 0.5, img, 0.5, 0)

    # Calculate the histogram of the image
    hist, _ = np.histogram(img, bins=256, range=(0, 256))

    # Find the minimum and maximum non-zero histogram bins
    min_bin, max_bin = np.argmax(hist > threshold), 255 - np.argmax(hist[::-1] > threshold)

    # Calculate alpha and beta for linear stretching
    alpha = 255.0 / max(1, max_bin - min_bin)
    beta = -min_bin * alpha

    # Apply gamma correction for enhancing darker and brighter regions
    enhanced_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return enhanced_img.astype(np.uint8)

def apply_mask(pil_image, binary_mask):
    # Convert Pillow image to NumPy array
    image_array = np.array(pil_image)

    # Ensure the binary mask is in uint8 format
    binary_mask = np.uint8(binary_mask)

    # Apply the binary mask to the image
    masked_image_array = cv2.bitwise_and(image_array, image_array, mask=binary_mask)

    # Convert the masked image array back to Pillow format
    masked_pil_image = Image.fromarray(masked_image_array)

    return masked_pil_image

def concat(image1, image2):
    """
    concat two images side by side.

    Parameters:
        image1 (PIL.Image): The first image (Pillow Image object).
        image2 (PIL.Image): The second image (Pillow Image object).

    Returns:
        PIL.Image: The combined image with both images side by side.
    """
    # Get the dimensions of the input images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Determine the combined width and height of the output image
    combined_width = width1 + width2
    combined_height = max(height1, height2)

    # Create a new blank image with the combined dimensions
    combined_image = Image.new('RGB', (combined_width, combined_height))

    # Paste the first image on the left side of the combined image
    combined_image.paste(image1, (0, 0))

    # Paste the second image on the right side of the combined image
    combined_image.paste(image2, (width1, 0))

    # Return the combined image
    return combined_image