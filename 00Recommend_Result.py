import json
import torch
import clip
import torch.nn.functional as F
import configparser
from openai import OpenAI
import pandas as pd
import numpy as np
from tqdm import tqdm

# 환경 설정 파일에서 API 키 읽기
config = configparser.ConfigParser()
config.read('config.ini')
api_key = config['openai']['api_key']

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

def cosine_similarity(vec1, vec2):
    """ 코사인 유사도를 수동으로 계산합니다. """
    dot_product = sum(a*b for a, b in zip(vec1, vec2))
    norm_a = sum(a*a for a in vec1) ** 0.5
    norm_b = sum(b*b for b in vec2) ** 0.5
    return dot_product / (norm_a * norm_b)


def euclidean_distance(vec1, vec2):
    """ 두 벡터 간의 유클리디안 거리를 계산합니다. """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.linalg.norm(vec1 - vec2)

with open("extracted_choice_features.json", "r", encoding='utf-8') as file:
     choice_features = json.load(file)

with open("place_text_features.json", "r", encoding='utf-8') as file:
      place_features = json.load(file)

def recommend_region(choice_embed, place_features):
    region_similarities = {}
    # 각 지역의 특성과 비교하여 유사도 계산
    for place in place_features:
        region_name = place['poi_region']
        region_features = place['embedding']
        similarity = cosine_similarity(choice_embed, region_features)
        region_similarities[region_name] = similarity

    # 유사도를 기준으로 가장 높은 지역을 추천
    recommended_region = sorted(region_similarities.items(), key=lambda x: x[1])[0][0]
    return recommended_region

# 결과 저장할 리스트
results = []

# 각 choice에 대해 추천 지역을 계산하고 결과를 저장
for choice in choice_features:
    choice_text = choice['choice']  # 이전에는 'choice' key를 찾을 수 없었습니다.
    choice_embed = choice['embedding']
    recommended_region = recommend_region(choice_embed, place_features)
    results.append({
        "choice": choice_text,
        "recommended_region": recommended_region
    })

# 결과를 JSON 파일로 저장
with open('C_recommendation_results.json', 'w', encoding='utf-8') as file:
    json.dump(results, file, ensure_ascii=False, indent=4)


# 추천받은 지역 내에서 여행지 5곳 추천받기
# 추천받은 여행지마다 음악 5개씩 추천받기
# 사용자 choice 마다 추천 결과 데이터 만들기
