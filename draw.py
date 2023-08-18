from PIL import Image, ImageDraw

__all__ = ["draw_bounding_boxes", "draw_binary_mask"]

# def draw_bounding_boxes(image, bounding_boxes, color=(0, 255, 0), thickness=2):
#     # Create a copy of the image to draw the bounding boxes
#     image_with_boxes = image.copy()

#     # Create a PIL ImageDraw object to draw on the image
#     draw = ImageDraw.Draw(image_with_boxes)

#     # Draw bounding boxes on the copied image
#     for bbox in bounding_boxes:
#         x1, y1, x2, y2 = bbox
#         draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

#     return image_with_boxes

def draw_bounding_boxes(image, labels, color=(0, 255, 0), thickness=2):
    # Create a copy of the image to draw the bounding boxes
    image_with_boxes = image.copy()

    # Get the image width and height
    img_width, img_height = image_with_boxes.size

    # Create a PIL ImageDraw object to draw on the image
    draw = ImageDraw.Draw(image_with_boxes)

    # Draw bounding boxes on the copied image
    for label in labels:
        x, y, w, h = label["bounding_box"]["coordinates"]

        # Convert normalized coordinates to image coordinates
        x_min = int(x * img_width)
        y_min = int(y * img_height)
        x_max = int((x + w) * img_width)
        y_max = int((y + h) * img_height)

        # Draw the bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=thickness)

        # Get the score and convert it to a string
        score = label["score"]
        score_str = "{:.2f}".format(score)

        # Calculate the position to draw the score text
        text_x = x_min
        text_y = y_min - 15  # Place the text above the bounding box

        # Draw the score text
        draw.text((text_x, text_y), score_str, fill=color)

    return image_with_boxes

def draw_binary_mask(image, mask, opacity=0.1, color=(0, 255, 0)):
    """
    Draw a binary mask on an image.

    Parameters:
        image (PIL.Image): The original image (Pillow Image object).
        mask (PIL.Image): The binary mask image (Pillow Image object). 
                          Should have the same size as the original image.
        opacity (float): The opacity of the mask. Default is 0.5.
        color (tuple): The RGB color tuple for the mask. Default is (0, 255, 0) (green).

    Returns:
        PIL.Image: The image with the binary mask applied.
    """
    # Make a copy of the original image to avoid modifying it directly
    image_copy = image.copy()

    # Create a new blank image for the mask overlay
    overlay = Image.new('RGBA', image_copy.size, (*color, int(255 * opacity)))

    # Paste the mask overlay using the binary mask as a transparency mask
    image_copy.paste(overlay, (0, 0), mask)

    return image_copy

