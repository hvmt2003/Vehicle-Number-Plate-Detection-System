# src/ocr/ocr.py
import easyocr
import cv2
import numpy as np
import re

# Initialize EasyOCR once
reader = easyocr.Reader(['en'], gpu=False)

def preprocess_plate(img):
    """
    img: BGR numpy array (crop)
    returns list of candidate preprocessed images (grayscale) to try OCR on.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) basic resized gray
    h, w = gray.shape
    scale = 2.5 if max(h,w) < 150 else 1.6
    small = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    # 2) denoised
    den = cv2.bilateralFilter(small, 9, 75, 75)

    # 3) adaptive threshold (good for uneven lighting)
    th = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 15, 6)

    # 4) Otsu after Gaussian blur
    blur = cv2.GaussianBlur(small, (3,3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5) morphological to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)

    # return candidates (order = prioritized)
    return [small, den, th, morph, otsu]

def ocr_on_image(img):
    """
    Try to get OCR text from an image (numpy gray or bgr).
    Returns (text, confidence_estimate)
    """
    # EasyOCR accepts grayscale or color arrays
    try:
        res = reader.readtext(img, detail=0)
    except Exception:
        # fallback to calling with converted type
        res = reader.readtext(img.astype('uint8'), detail=0)

    if not res:
        return None, 0.0
    # join all pieces
    text = "".join(res).strip()
    # simple confidence estimate: length of result and char types
    score = len(text)
    return text, float(score)

def extract_text_from_crop(crop_bgr):
    """
    Accepts BGR crop numpy image, returns best cleaned & formatted plate text.
    """
    candidates = preprocess_plate(crop_bgr)
    best_text = None
    best_score = -1

    # Try OCR on each candidate and keep best by heuristic score
    for ci in candidates:
        text, score = ocr_on_image(ci)
        if text:
            clean = clean_plate_text(text)
            # scoring: prefer results with length >=6 and more digits in numeric part
            digit_count = sum(c.isdigit() for c in clean) if clean else 0
            score_adj = score + (digit_count * 2)
            if score_adj > best_score:
                best_score = score_adj
                best_text = clean

    # as a last resort try OCR on color image
    if best_text is None:
        text, score = ocr_on_image(crop_bgr)
        best_text = clean_plate_text(text) if text else None

    formatted = format_plate(best_text)
    return best_text, formatted, best_score

import re

def clean_plate_text(text):
    if text is None:
        return None

    text = text.upper()

    # Remove unwanted
    text = re.sub(r"[^A-Z0-9]", "", text)

    # COMMON OCR MISTAKES (STRONG RULES)
    corrections = {
        "H": "M",   # H often mistaken for M
        "O": "0",   # O → 0
        "I": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6"
    }

    # apply corrections
    corrected = ""
    for ch in text:
        corrected += corrections.get(ch, ch)

    text = corrected

    # --- APPLY INDIAN PLATE STRUCTURE CORRECTIONS ---
    # Expected format: LL NN LL NNNN

    # Fix first two letters (state code)
    if len(text) >= 2:
        if not text[0].isalpha():
            text = "M" + text[1:]
        if not text[1].isalpha():
            text = text[0] + "H"

    # Fix district code (2 digits)
    if len(text) >= 4:
        sec = text[2:4]
        sec = sec.replace("O", "0").replace("D", "0")
        if not sec.isdigit():
            sec = re.sub(r"[A-Z]", "0", sec)
        text = text[:2] + sec + text[4:]

    # Fix series (letters)
    if len(text) >= 6:
        series = text[4:6]
        series = series.replace("2", "Z")   # if mistakenly numeric
        # but enforce letters
        series = re.sub(r"[0-9]", "V", series) 
        text = text[:4] + series + text[6:]

    return text


def format_plate(text):
    if not text:
        return None
    text = text.upper()
    # if short, return as-is
    if len(text) < 6:
        return text
    # split into groups
    state = text[0:2]
    district = text[2:4]
    series = text[4:6]
    number = text[6:]
    return f"{state} {district} {series} {number}"
