import numpy as np
import cv2
from PIL import Image

__all__ = ["apply_mask", "concat"]

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