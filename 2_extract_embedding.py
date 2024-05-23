import os
import torch
import clip
from PIL import Image
import json
from tqdm import tqdm
import numpy as np
import csv

# 사용할 디바이스를 'cpu'로 설정
DEVICE = "cpu"
MODEL_NAME = "ViT-B/32"
# CLIP 모델을 불러옵니다. 이 모델은 이미지와 텍스트의 특징을 추출할 수 있습니다.
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)


# 이미지 특징과 지역명 이름도 함께 저장하는것으로 수정

# def extract_place_features():
#     # 이미지에서 특징을 추출하여 JSON 파일로 저장하는 함수
#     features = {}
#     images_dir = os.path.join(os.getcwd(), "images")
#
#     # 이미지 디렉토리에서 모든 파일을 순회
#     for image_name in tqdm(os.listdir(images_dir)):
#         image_path = os.path.join(images_dir, image_name)
#         try:
#             # 이미지를 불러오고, 전처리한 후 모델에 입력할 수 있는 형태로 만듭니다.
#             image = preprocess(Image.open(image_path)).unsqueeze(0).to("cpu")
#         except:
#             print(f'Error: {image_name}')
#             continue
#
#         with torch.no_grad():
#             # 이미지에서 특징을 추출
#             image_features = model.encode_image(image)
#             features[image_name] = image_features.cpu().numpy().tolist()
#
#     # 특징을 JSON 파일로 저장
#     with open("place_features.json", "w", encoding='utf-8') as fp:
#         json.dump(features, fp)


def extract_music_features():
    # 음악 가사에서 특징을 추출하여 JSON 파일로 저장하는 함수
    features = {}
    with open('Music_info.csv', 'r', encoding='utf-8') as file:
        # csv.reader를 사용해 파일 읽기
        csv_reader = csv.reader(file, delimiter=',', quotechar='"')
        next(csv_reader)  # 헤더 스킵

        for row in tqdm(csv_reader):
            if len(row) == 5:  # 올바른 필드 수 확인
                genre, title, singer, lyrics, image = row

                # 가사를 40자 길이로 나누어 각 부분의 임베딩을 계산
                text_embeddings = []
                for i in range(0, len(lyrics), 30):
                    text = clip.tokenize([lyrics[i:i + 30]]).to('cpu')
                    with torch.no_grad():
                        text_features = model.encode_text(text)
                        text_embeddings.append(text_features.cpu().numpy())

                if text_embeddings:
                    mean_embedding = np.mean(text_embeddings, axis=0)
                    features[f'{singer}-{title}'] = {
                        'genre': genre,
                        'embedding': mean_embedding.tolist(),
                        'image': image
                    }
            else:
                print(f"Skipping row due to unexpected format: {row}")

    with open("music_features.json", "w", encoding='utf-8') as fp:
            json.dump(features, fp, ensure_ascii=False, indent=4)

    print("Features extracted and saved to 'music_features.json'")





def chech_feature_counts():
    # 저장된 특징의 개수를 확인하는 함수
    with open("music_features.json", "r", encoding='utf-8') as fp:
        features = json.load(fp)

    for key, value in features.items():
        print(key)
    print(len(features))


# if __name__ == "__main__":
#     extract_place_features()  # 이미지 특징 추출 함수 호출
    extract_music_features()  # 음악 특징 추출 함수 호출
#     chech_feature_counts()  # 특징 개수 확인 함수 호출
