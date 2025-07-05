import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# ----- Paths -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_SAFE = os.path.join(BASE_DIR, "data", "phonesafe")
RAW_BROKEN = os.path.join(BASE_DIR, "data", "phonebroken")
PREP_DIR = os.path.join(BASE_DIR, "data", "prepared")
MODEL_PATH = os.path.join(BASE_DIR, "iphone_classifier.pth")
# ------------------

IMG_SIZE = (224, 224)
EPOCHS = 5
BATCH_SIZE = 32

# Step 1: Prepare dataset
def prepare_dataset():
    print("📁 Preparing dataset folders...")
    SAFE_DIR = os.path.join(PREP_DIR, "safe")
    BROKEN_DIR = os.path.join(PREP_DIR, "broken")

    # Remove old prepared data
    if os.path.exists(PREP_DIR):
        shutil.rmtree(PREP_DIR)
    os.makedirs(SAFE_DIR, exist_ok=True)
    os.makedirs(BROKEN_DIR, exist_ok=True)

    def copy_and_resize(src, dst):
        for filename in os.listdir(src):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    img = Image.open(os.path.join(src, filename)).convert("RGB")
                    img = img.resize(IMG_SIZE)
                    img.save(os.path.join(dst, filename), "JPEG")
                except Exception as e:
                    print(f"⚠️ Skipped {filename}: {e}")

    copy_and_resize(RAW_SAFE, SAFE_DIR)
    copy_and_resize(RAW_BROKEN, BROKEN_DIR)
    print("✅ Dataset prepared!")

# Step 2: Train model
def train_model():
    print("🚀 Training model...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(PREP_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"📊 Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

# Run all
if __name__ == "__main__":
    prepare_dataset()
    train_model()
