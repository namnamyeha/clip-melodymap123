import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
import clip

# CLIP 모델과 전처리 함수 설정
model, preprocess = clip.load("ViT-B/32", device="cpu")

def extract_place_features():
    # 이미지에서 특징을 추출하여 JSON 파일로 저장하는 함수
    features = []
    images_dir = os.path.join(os.getcwd(), "images")

    # 이미지 디렉토리에서 모든 파일을 순회
    for image_name in tqdm(os.listdir(images_dir)):
        if image_name.endswith('.jpg'):  # JPG 이미지만 처리
            region_name = image_name[:2]  # 파일 이름에서 지역 이름 추출
            place_name = image_name[2:-4]  # 파일 이름에서 여행지 이름 추출 (확장자 제외)

            image_path = os.path.join(images_dir, image_name)
            try:
                # 이미지를 불러오고, 전처리한 후 모델에 입력할 수 있는 형태로 만듭니다.
                image = preprocess(Image.open(image_path)).unsqueeze(0).to("cpu")
            except Exception as e:
                print(f'Error processing {image_name}: {str(e)}')
                continue

            with torch.no_grad():
                # 이미지에서 특징을 추출
                image_features = model.encode_image(image)
                features.append({
                    "region": region_name,
                    "place": place_name,
                    "embedding": image_features.cpu().numpy().tolist()
                })

    # 특징을 JSON 파일로 저장
    with open("place_features_img.json", "w", encoding='utf-8') as fp:
        json.dump(features, fp, ensure_ascii=False, indent=4)

# 함수 호출
extract_place_features()
