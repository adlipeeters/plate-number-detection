import os
from google.cloud import vision

# Assuming your JSON key is uploaded to your Colab root and named 'vision_key.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision_key.json'

from google.cloud import vision


def detect_text(path):
    """
    Extracts text from an image using Google Cloud Vision API.

    Args:
        path: Path to the image file.

    Returns:
        String containing the extracted text.
    """
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message))

    return texts[0].description

    # Assuming your image is located in 'images/image1.jpg'


image_path = 'images/2.jpg'

text = detect_text(image_path)

print(f"Extracted Text: \n{text}")

