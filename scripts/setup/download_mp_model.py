import requests
import os

url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
output = "pose_landmarker_lite.task"

if not os.path.exists(output):
    print(f"Downloading {output}...")
    response = requests.get(url)
    with open(output, 'wb') as f:
        f.write(response.content)
    print("Download complete.")
else:
    print("File already exists.")
