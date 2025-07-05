import cv2
import numpy as np
from paddleocr import PaddleOCR

# ✅ Initialize once for reuse
ocr_model = PaddleOCR(lang='en', use_angle_cls=True)

def deskew(image: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(image > 0))
    if coords.shape[0] == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_for_ocr(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    deskewed = deskew(clahe)
    thresh = cv2.adaptiveThreshold(deskewed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    inverted = cv2.bitwise_not(closed)
    sharpened = cv2.filter2D(inverted, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

def get_logo_text(image_path: str):
    processed_img = preprocess_for_ocr(image_path)
    results = ocr_model.predict(processed_img)

    text_blocks = []
    for res in results:
        for txt, score in zip(res.get('rec_texts', []), res.get('rec_scores', [])):
            text_blocks.append((txt, score))
    return text_blocks

import cv2

def run_logo_ocr(image_path: str):
    print(f"🧾 Running OCR on: {image_path}")
    try:
        processed_img = preprocess_for_ocr(image_path)

        # 👀 Show the preprocessed image
        cv2.imshow("Preprocessed Image", processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 💾 Optionally save it
        cv2.imwrite("preprocessed_debug.jpg", processed_img)

        results = ocr_model.predict(processed_img)
    except Exception as e:
        print(f"❌ OCR failed: {e}")
        return []

    text_blocks = []
    for res in results:
        for txt, score in zip(res.get('rec_texts', []), res.get('rec_scores', [])):
            text_blocks.append((txt, score))

    if not text_blocks:
        print("⚠️ No text detected.")
        return []

    print("\n🔍 Detected Text:")
    for txt, score in text_blocks:
        print(f"→ {txt} (Confidence: {score:.2f})")

    return text_blocks


def main():
    image_path = r'C:\Users\Prajanya\Desktop\wallmartHack\ai_model\logo_detector\testimg.jpg'
    run_logo_ocr(image_path)

if __name__ == "__main__":
    main()
