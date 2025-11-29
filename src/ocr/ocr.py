import easyocr
import cv2
import numpy as np

# Initialize EasyOCR reader (English)
reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_plate(plate_image_path):
    """
    Runs OCR on a cropped plate image.
    Converts to grayscale to avoid shape mismatch errors.
    """
    img = cv2.imread(plate_image_path)

    if img is None:
        print("Error: Could not read cropped plate image.")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Run OCR
    results = reader.readtext(gray, detail=0)

    if len(results) == 0:
        return None

    # Cleanup output
    text = "".join(results).replace(" ", "").upper()
    return text

import re

def clean_plate_text(text):
    """
    Cleans OCR output and corrects common mistakes for Indian number plates.
    Format: AA NN AA NNNN
    Example: MH02EU1884
    """

    if text is None:
        return None

    # Replace common OCR mistakes
    text = text.replace("O", "0")   # O → 0
    text = text.replace("I", "1")   # I → 1
    text = text.replace("Z", "2")   # Z → 2

    # Keep only alphanumeric characters
    text = re.sub(r"[^A-Za-z0-9]", "", text)

    # Uppercase
    text = text.upper()

    return text

def format_plate(text):
    """
    Format cleaned OCR output into official Indian number plate pattern:
    AA 00 AA 0000
    """
    if text is None or len(text) < 6:
        return text

    # Fix common OCR mistakes
    text = text.replace("L", "1")  # L misread instead of 1

    # Ensure uppercase
    text = text.upper()

    # Try to split according to pattern
    try:
        state = text[0:2]          # MH
        district = text[2:4]       # 02
        series = text[4:6]         # EU
        number = text[6:]          # 1884

        formatted = f"{state} {district} {series} {number}"
        return formatted

    except:
        return text
