import json
import torch
import clip
import torch.nn.functional as F
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
    # 각 지역의 특성과 비교하여 유사도 계산 (코사인 유사도 사용)
    for feature in place_features:
        region_name = feature['region']  # 각 항목에서 지역 이름 접근
        region_features = torch.tensor(feature['features']).unsqueeze(0).to(DEVICE)
        similarity = F.cosine_similarity(text_features, region_features).mean().item()
        region_similarities[region_name] = similarity

    # 유사도가 높은 순으로 정렬하여 가장 관련성 높은 지역 추천
    recommended_region = sorted(region_similarities.items(), key=lambda x: x[1], reverse=True)[0][0]
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

def recommend_music(place_name: str) -> str:

    # 지정된 장소 이름에 대한 음악 추천
    with open("place_features.json", "r", encoding='utf-8') as fp:
        place_features = json.load(fp)
    with open("music_features.json", "r", encoding='utf-8') as fp:
        music_features = json.load(fp)

        # 가장 유사한 장소 찾기
        keyword_features = model.encode_text(clip.tokenize([place_name]).to(DEVICE)).float()
        best_match = None
        best_similarity = float('inf')
        for place in place_features:
            features = torch.tensor(place['features'])
            similarity = torch.nn.functional.pairwise_distance(keyword_features.unsqueeze(0),
                                                               features.unsqueeze(0)).sum().item()
            if similarity < best_similarity:
                best_similarity = similarity
                best_match = place['place']

        if best_match is None:
            return "장소를 찾을 수 없습니다."

        # 해당 장소의 특징
        place_feature = next((place['features'] for place in place_features if place['place'] == best_match), None)

        # 음악 특징과 장소 특징 간 유사도 계산(거리가 짧을수록 유사도가 높다)
        similarities = {}
        for music_name, details in music_features.items():
            music_feature = torch.tensor(details['embedding'])
            similarity = torch.nn.functional.pairwise_distance(torch.tensor(place_feature), music_feature).sum().item()
            similarities[music_name] = similarity

        # 유사도가 높은 순으로 정렬하여 상위 5개 추천
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1])
        return sorted_similarities[2:8]


# JSON 파일을 불러오기
with open('melodymap_modify.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 새로운 결과를 저장할 리스트
new_results = []

# 데이터 파일의 모든 항목에 대해 처리
for i in range(len(data)):  # 이 부분을 수정하여 전체 데이터를 처리
    recommand_keyword = data[i]['Recommand_keyword']
    recommand_choice = data[i]['Recommand_choice']  # 'Recommand_choice' 추출

    # 지역 추천
    recommended_region = recommend_region(recommand_keyword)

    # 지역 내 장소 추천
    recommended_places = recommend_places_in_region(recommended_region)

    # 추천 음악
    recommended_music = recommend_music(recommand_keyword)

    # 결과를 새로운 딕셔너리로 저장
    result_entry = {
        "Recommand_choice": recommand_choice,  # 'Recommand_choice'를 결과에 추가
        "Recommand_keyword": recommand_keyword,
        "Recommended_region": recommended_region,
        "Recommended_places": recommended_places,
        "Recommended_music": recommended_music
    }
    new_results.append(result_entry)

    # 결과 출력
    print(f"#{i + 1} 처리 결과:")
    print(f"추천 유형: {recommand_choice}")
    print(f"추천 키워드: {recommand_keyword}")
    print(f"추천 지역: {recommended_region}")
    print(f"추천 장소: {recommended_places}")
    print(f"추천 음악: {recommended_music}")
    print("\n")

# 모든 결과를 JSON 파일에 저장
with open('melodymap_results.json', 'w', encoding='utf-8') as file:
    json.dump(new_results, file, ensure_ascii=False, indent=2)