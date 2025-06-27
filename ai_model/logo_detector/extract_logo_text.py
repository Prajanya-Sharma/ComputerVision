import cv2
import numpy as np
from paddleocr import PaddleOCR

def deskew(image: np.ndarray) -> np.ndarray:
    """
    Compute the skew angle of a binarized image and rotate to correct it.
    If no foreground pixels are found, returns the original image.
    """
    coords = np.column_stack(np.where(image > 0))
    if coords.shape[0] == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

def preprocess_for_ocr(image_path: str) -> np.ndarray:
    """
    Load an image, enhance it for OCR, and return a 3-channel BGR image ready for PaddleOCR.
    """
    # 1. Load & upscale
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Noise reduction
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 4. Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 5. Deskewing
    deskewed = deskew(enhanced)

    # 6. Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        deskewed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=3
    )

    # 7. Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 8. Invert
    inverted = cv2.bitwise_not(closed)

    # 9. Sharpen
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(inverted, -1, kernel_sharp)

    # 10. Convert to BGR for PaddleOCR
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

def get_logo_text(image_path: str):
    """
    Use PaddleOCR to extract text from a processed logo image.
    Returns a list of (text, confidence) tuples.
    """
    ocr = PaddleOCR(lang='en', use_angle_cls=True)
    processed_img = preprocess_for_ocr(image_path)
    results = ocr.predict(processed_img)

    text_blocks = []
    for res in results:
        texts = res.get('rec_texts', [])
        scores = res.get('rec_scores', [])
        for txt, sc in zip(texts, scores):
            text_blocks.append((txt, sc))
    return text_blocks

def main():
    image_path = r'C:\Users\Prajanya\Desktop\wallmartHack\ai_model\logo_detector\testimg.jpg'
    text_blocks = get_logo_text(image_path)

    print("\n🧾 Detected Text:")
    for txt, score in text_blocks:
        print(f"{txt} (Confidence: {score:.2f})")

if __name__ == "__main__":
    main()
