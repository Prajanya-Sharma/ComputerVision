import os
import requests
import csv
from dotenv import load_dotenv
from groq import Groq
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

def resize_image_if_needed(image_path, max_size=1024):
    img = Image.open(image_path)
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))
        img.save(image_path)

def upload_image_to_imgbb(image_path):
    with open(image_path, "rb") as f:
        response = requests.post(
            "https://api.imgbb.com/1/upload",
            params={"key": IMGBB_API_KEY},
            files={"image": f}
        )
    response.raise_for_status()
    return response.json()["data"]["url"]


def analyze_fabric_type(image_url):
    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Fabric type? (One word only)"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=10,
            top_p=1,
            stream=False,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f" Error analyzing image: {e}"

def process_image(image_path):
    try:
        print(f"📤 Uploading {image_path}...")
        resize_image_if_needed(image_path)
        image_url = upload_image_to_imgbb(image_path)
        print(f"✅ Uploaded: {image_url}")
        result = analyze_fabric_type(image_url)
        return (image_path, result)
    except Exception as e:
        return (image_path, f" Failed: {e}")

MAX_RETRIES = 3

def run_with_retries(image_paths):
    attempts = 0
    results = []
    retry_queue = image_paths.copy()

    while retry_queue and attempts < MAX_RETRIES:
        print(f"\n🔁 Retry Attempt {attempts + 1}...\n")
        next_retry = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_image, path): path for path in retry_queue}
            for future in as_completed(futures):
                image_path, result = future.result()
                if "❌" in result:
                    next_retry.append(image_path)
                else:
                    print(f"\n🧶 Fabric result for {image_path}: {result}")
                    results.append((image_path, result))

        retry_queue = next_retry
        attempts += 1

    for path in retry_queue:
        results.append((path, "❌ Failed after retries"))

    return results

def main():
    image_files = [
        "ai_model/deepseek_vl/testimg.jpg",
        "ai_model/deepseek_vl/testimg2.jpg",
        "ai_model/deepseek_vl/testimg3.jpg",
        "ai_model/deepseek_vl/testimg4.jpg",
    ]

    print("🔎 Detecting fabric types with retry queue...\n")
    results = run_with_retries(image_files)

    with open("fabric_results.csv", "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image", "FabricType"])
        writer.writerows(results)

    print("\n✅ Results saved to fabric_results.csv")

# ⛔ Prevent auto-run if imported
if __name__ == "__main__":
    main()