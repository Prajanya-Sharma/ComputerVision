import sys
import os

# Add parent directory to sys.path to import from ai_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_model.deepseek_vl import gemini_image_infer
from ai_model.fabric_classifier import predict_fabric
from ai_model.logo_detector.extract_logo_text import run_logo_ocr


def run_inference():
    print("🚀 Running fabric analysis from main app...\n")
    gemini_image_infer.main()

def run_fabric_prediction():
    print("🔍 Running fabric prediction from main app...\n")
    folder = "test_uploads"
    image_filename = "lemmetest.jpg"

    try:
        top_label = predict_fabric.run_prediction(folder, image_filename)
        print(f"\n🏷️ Predicted Fabric Label: {top_label}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

def run_logo_text_ocr():
    print("🧾 Running logo text OCR from main app...\n")
    image_path = os.path.join("test_uploads", "imt.jpg")
    
    try:
        run_logo_ocr(image_path)  # ✅ Correct: directly use the imported function
    except Exception as e:
        print(f"❌ Logo OCR failed: {e}")


if __name__ == "__main__":
    #run_inference()
    #run_fabric_prediction()
    run_logo_text_ocr()
