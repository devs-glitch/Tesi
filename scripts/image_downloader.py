''' Create class folders and dowload spectrogramns images from urls'''

import pandas as pd
import os
import requests

base_dir = "data/processed/"
selected_df = pd.read_csv("data/processed/selected_dataset.csv")
classes = selected_df['ml_label'].unique()

for i in classes:
    class_path = os.path.join(base_dir, i)
    os.makedirs(class_path, exist_ok=True)

for index, row in selected_df.iterrows():
    curr_class = row['ml_label']
    image_url = row['url1']
    file_name = f"glitch_{index}.png"
    saved_path= os.path.join(base_dir, curr_class, file_name)

    if os.path.exists(saved_path):
        continue

    try:
        response = requests.get(image_url, timeout=10)
        with open(saved_path, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        print(f"Error download row {index}: {e}")

    if index % 1000 == 0:
        print(f"Downloading...: current image n° {index}...")
