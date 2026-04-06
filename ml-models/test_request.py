import base64
import requests

with open("test.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = {
    "model_id": "2ff57b6c",
    "input_data": {
        "image_b64": img_b64
    }
}

res = requests.post("http://localhost:8002/infer", json=payload)

print("Status:", res.status_code)
print("Response:", res.json())