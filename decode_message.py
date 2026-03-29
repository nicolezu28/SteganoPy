def decode_message_classic(image_path, delimiter='|', min_length=10):
    import numpy as np
    from PIL import Image

    # Load image
    image = Image.open(image_path)
    pixels = np.array(image)
    message = ''
    for pixel in pixels:
        # Collect the least significant bit
        for color in pixel:
            message += str(color & 1)

    # Check for minimum message length before delimiter check
    if len(message) < min_length:
        return None  # or handle as needed

    # Split message using delimiter
    parts = message.split(delimiter)
    # Verify delimiter logic.
    if len(parts) < 2:
        return None  # or handle as needed

    return ''.join(parts[:-1])  # Return the concatenated message without the last part