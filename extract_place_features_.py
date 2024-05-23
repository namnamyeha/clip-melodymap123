import json
import torch
import clip

# CLIP 모델 로드
DEVICE = "cpu"
MODEL_NAME = "ViT-B/32"
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)

def recommend_region(keyword: str) -> str:
    # place_features.json에서 지역 특성 로드
    with open("place_features.json", "r", encoding='utf-8') as fp:
        place_features = json.load(fp)

    # 키워드로부터 특성 벡터 추출
    text = clip.tokenize([keyword]).to(DEVICE)
    text_features = model.encode_text(text)

    region_similarities = {}
    # 각 지역의 특성과 비교하여 유사도 계산
    for feature in place_features:
        region_name = feature['region']  # 각 항목에서 지역 이름 접근
        region_features = torch.tensor(feature['features']).unsqueeze(0).to(DEVICE)
        similarity = torch.nn.functional.pairwise_distance(
            text_features, region_features).sum().item()
        region_similarities[region_name] = similarity

    # 유사도가 낮은 순으로 정렬하여 가장 관련성 높은 지역 추천
    recommended_region = sorted(region_similarities.items(), key=lambda x: x[1])[0][0]
    return recommended_region

def recommend_places_in_region(region: str) -> list:
    with open("place_features.json", "r", encoding='utf-8') as fp:
        place_features = json.load(fp)

    place_similarities = {}
    for feature in place_features:
        if feature['region'] == region:
            place_name = feature['place']  # 여행지 이름
            similarity = torch.nn.functional.pairwise_distance(
                torch.tensor(feature['features']), torch.tensor([0]*len(feature['features']))).sum().item()
            place_similarities[place_name] = similarity


    recommended_places = sorted(place_similarities.items(), key=lambda x: x[1])[:5]
    return recommended_places

# 사용자의 결과값 불러오기
with open('melodymap_modify.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 첫 번째 사용자 결과의 키워드 사용
recommand_keyword = data[0]['Recommand_keyword']

# 지역 추천
recommended_region = recommend_region(recommand_keyword)
print("추천 지역:", recommended_region)

# 지역 내 장소 추천
recommended_places = recommend_places_in_region(recommended_region)
print("추천 장소:", recommended_places)
