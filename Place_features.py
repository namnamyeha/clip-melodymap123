import os
import json
from PIL import Image
import torch
import clip
import requests
from io import BytesIO
from tqdm import tqdm

# CLIP 모델 로드
model, preprocess = clip.load("ViT-B/32", device="cpu")

def download_image_from_url(url):
    """ URL에서 이미지를 다운로드하고 PIL 이미지 객체로 반환합니다. """
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def extract_combined_features():
    """ JSON 파일에서 이미지 URL과 텍스트 데이터를 읽어 특성을 추출합니다. """
    with open("Travle/Travel_pre.json", "r", encoding='utf-8') as file:
        data = json.load(file)

    combined_features = []

    for item in tqdm(data):
        try:
            region = item['poi_region']
            place = item['poi_name']
            image_url = item['img_rname']  # 이미지 URL

            # URL에서 이미지 다운로드
            image = download_image_from_url(image_url)
            image = preprocess(image).unsqueeze(0).to("cpu")

            with torch.no_grad():
                # 이미지에서 특성 추출
                image_features = model.encode_image(image).cpu().numpy()

                # 특성이 2차원인지 확인
                if image_features.ndim == 2:
                    image_features = image_features.flatten()

                combined_features.append({
                    "region": region,
                    "place": place,
                    "embedding": image_features.tolist()
                })
        except Exception as e:
            print(f"Error processing image from {place}: {e}")
            continue

    # 추출된 특징을 JSON 파일로 저장
    with open("place_image_features.json", "w", encoding='utf-8') as fp:
        json.dump(combined_features, fp, ensure_ascii=False, indent=4)

# 함수 호출
extract_combined_features()