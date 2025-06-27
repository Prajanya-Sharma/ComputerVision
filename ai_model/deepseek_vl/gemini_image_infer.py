import os
import time
import requests
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env
load_dotenv()

# API Keys
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def upload_image(image_path):
    with open(image_path, "rb") as file:
        res = requests.post(
            "https://api.imgbb.com/1/upload",
            params={"key": IMGBB_API_KEY},
            files={"image": file}
        )
    res.raise_for_status()
    return res.json()["data"]["url"]

def query_gemini(image_url):
    retries = 3
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="google/gemini-2.0-flash-exp:free",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this image?"},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                extra_headers={
                    "HTTP-Referer": os.getenv("YOUR_SITE_URL"),
                    "X-Title": os.getenv("YOUR_SITE_NAME"),
                }
            )
            return completion.choices[0].message.content
        except RateLimitError:
            time.sleep(5)
        except Exception as e:
            return f"❌ Error processing image: {e}"
    return "❌ Failed after retries."

def process_image(image_path):
    try:
        print(f"📤 Uploading: {image_path}")
        image_url = upload_image(image_path)
        print(f"✅ Uploaded: {image_url}")
        response = query_gemini(image_url)
        return (image_path, response)
    except Exception as e:
        return (image_path, f"❌ Error: {e}")

if __name__ == "__main__":
    image_paths = [
        "ai_model/deepseek_vl/testimg.jpg",
        "ai_model/deepseek_vl/testimg2.jpg",
        "ai_model/deepseek_vl/testimg3.jpg",
        "ai_model/deepseek_vl/testimg4.jpg",
    ]

    print("🔄 Processing multiple images...\n")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_image, path) for path in image_paths]

        for future in as_completed(futures):
            img_path, output = future.result()
            print(f"\n🖼️ Result for {img_path}:\n{output}\n")
