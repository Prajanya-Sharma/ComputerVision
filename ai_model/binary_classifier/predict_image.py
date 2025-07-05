import os
import torch
from torchvision import models, transforms
from PIL import Image

# ----- Paths -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "iphone_classifier.pth")
TEST_IMAGE_PATH = os.path.join(ROOT_DIR, "test_uploads", "z.jpg")
# ------------------

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()

    label = "BROKEN" if pred_class == 0 else "SAFE"
    print(f"🔍 Prediction: {label} ({confidence * 100:.2f}% confidence)")

# Run prediction
if __name__ == "__main__":
    predict(TEST_IMAGE_PATH)
