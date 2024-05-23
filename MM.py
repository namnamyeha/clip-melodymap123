import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import clip
from PIL import Image
import requests
from io import BytesIO
import json
import torch.nn.functional as F

# CLIP 모델 로드
DEVICE = "cpu"
MODEL_NAME = "ViT-B/32"
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)

# 예시 데이터

with open('extracted_choice_features.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

df = pd.DataFrame(data)

# 텍스트 특성으로 지역추천 받기
def reco_region_text(choice: str, embedding: list) -> str:
    with open("extracted_place_features_1.json", "r", encoding='utf-8') as fp:
        place_feature = json.load(fp)
        text_features = torch.tensor(embedding).unsqueeze(0).to(DEVICE)
        region_similarities = {}
        for feature in place_feature:
            region = feature['region']  # 지역이름
            place = feature['place']  # 장소이름
            region_features = torch.tensor(feature['embedding']).unsqueeze(0).to(DEVICE)
            similarity = F.cosine_similarity(text_features, region_features).mean().item()
            region_similarities[region] = similarity
        # 유사도 정렬하여 관련성 높은 지역 추천
        recommended_region = max(region_similarities, key=region_similarities.get)
        print(f"사용자 유형 '{choice}': 추천지역은 '{recommended_region}'.")
    return recommended_region

# 이미지 특성으로 지역 추천 받기
def reco_region_img(choice:str, embedding:list) -> str:
    with open("place_image_features.json", "r", encoding='utf-8') as fp:
        place_feature = json.load(fp)
        text_features = torch.tensor(embedding).unsqueeze(0).to(DEVICE)
        region_similarities = {}

        for feature in place_feature:
            region = feature['region']  # 지역이름
            place = feature['place']  # 장소이름
            region_features = torch.tensor(feature['embedding']).unsqueeze(0).to(DEVICE)
            similarity = F.cosine_similarity(text_features, region_features).mean().item()
            region_similarities[region] = similarity
        recommended_region = max(region_similarities, key=region_similarities.get)
        print(f"사용자 유형 '{choice}': 추천지역은 '{recommended_region}'.")
    return recommended_region

#초이스 - 이미지, 초이스 - 텍스트 결과

df['image_features'] = df['embedding'].apply(reco_region_text())
df['text_features'] = df['embedding'].apply(reco_region_text())

# 특성과 타겟 분리
X_image = np.stack(df['image_features'].values)
X_text = np.stack(df['text_features'].values)
y = df['rating']

# 훈련 데이터와 테스트 데이터로 분할
X_image_train, X_image_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_image, X_text, y, test_size=0.2, random_state=42
)

# 기본 모델 학습 (이미지 특성)
model_image = RandomForestRegressor(n_estimators=100, random_state=42)
model_image.fit(X_image_train, y_train)
pred_image_train = model_image.predict(X_image_train)
pred_image_test = model_image.predict(X_image_test)

# 기본 모델 학습 (텍스트 특성)
model_text = RandomForestRegressor(n_estimators=100, random_state=42)
model_text.fit(X_text_train, y_train)
pred_text_train = model_text.predict(X_text_train)
pred_text_test = model_text.predict(X_text_test)
