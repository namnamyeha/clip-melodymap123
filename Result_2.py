import json
import torch
import clip
import torch.nn.functional as F
import configparser
from openai import OpenAI
import pandas as pd
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

with open("extracted_choice_features.json", "r", encoding='utf-8') as file:
     choice_features = json.load(file)

def recommend_region(choice: str) :
    # 임베딩된 장소 특성 로드
    with open("extracted_place_features.json", "r", encoding='utf-8') as file:
        place_features = json.load(file)


    region_similarities = {}

        # 각 지역의 특성과 비교하여 유사도 계산 (코사인 유사도 사용)
    for place, choice in tqdm(zip(place_features, choice_features)):
        region_name = place['region'] # 각 항목에서 지역 이름 접근
        region_features = place['embedding']
        choice_embed = choice['embedding']
        weight = choice.get('weight',1)

            # 코사인 유사도 수동계산
        similarity = cosine_similarity(choice_embed, region_features)
            # region_similarities[region_name] = similarity
        # 유사도에 가중치 적용
        weithed_smilarity = similarity * weight

        region_similarities[region_name] = weithed_smilarity

     # 유사도 정렬하여 관련성 높은 지역 추천
        recommended_region = sorted(region_similarities.items(), key=lambda x: x[1], reverse=True)[0][0]
        print(f"사용자 유형 '{choice['choice']}의 ': 추천지역은 '{recommended_region}'.")
    return recommended_region

for i in choice_features:
    recommend_region(i['choice'])


def recommend_places_in_region(region: str) -> list:
    with open("extracted_place_features.json", "r", encoding='utf-8') as fp:
        place_features = json.load(fp)
    place_similarities = {}
    for feature in place_features:
        if feature['region'] == region:
            place_name = feature['place']
            similarity = torch.nn.functional.pairwise_distance(torch.tensor(feature['embedding']), torch.tensor([0]*len(feature['features']))).sum().item()
            place_similarities[place_name] = similarity
    recommended_places = sorted(place_similarities.items(), key=lambda x: x[1])[:5]
    print(f"지역에 따른 '{region}': 추천여행지는 {[place for place, _ in recommended_places]}.")
    return recommended_places

def recommend_music(place_name: str) -> list:
    with open("extracted_place_features.json", "r", encoding='utf-8') as fp:
        place_features = json.load(fp)
    with open("e_music_features.json", "r", encoding='utf-8') as fp:
        music_features = json.load(fp)

    place_feature = next((place['embedding'] for place in place_features if place['place'] == place_name), None)
    similarities = {}
    for music in music_features:
        music_feature = torch.tensor(music['embedding'])
        similarity = torch.nn.functional.pairwise_distance(torch.tensor(place_feature), music_feature).sum().item()
        similarities[music['music']] = similarity
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1])[:5]
    recommended_music = [music for music, _ in sorted_similarities]
    print(f"장소 '{place_name}': 추천 음악 {recommended_music}.")
    return recommended_music
#
# def process_all_recommendations(data):
#     new_results = []
#     for item in data:
#         recommand_choice = item['Recommand_choice']
#         recommand_keyword = item['Recommand_keyword']
#         # 함수 호출 시 두 개의 필요한 인자 모두 전달
#         recommended_region = recommend_region(recommand_choice, recommand_keyword)
#         recommended_places = recommend_places_in_region(recommended_region)
#         place_music_map = {}
#         for place, _ in recommended_places:
#             music_list = recommend_music(place)
#             place_music_map[place] = music_list
#         new_results.append({
#             "Recommand_choice": recommand_choice,
#             "Recommand_keyword": recommand_keyword,
#             "Recommended_region": recommended_region,
#             "Place_to_music_map": place_music_map
#         })
#     with open('melodymap_results3.json', 'w', encoding='utf-8') as file:
#         json.dump(new_results, file, ensure_ascii=False, indent=2)
#     print("All recommendations have been processed and saved.")
#     return new_results
#
# # 데이터 불러오기
# with open('melodymap_modify.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)
#
# # 전체 추천 과정 실행
# final_results = process_all_recommendations(data)
