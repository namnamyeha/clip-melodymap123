import json
import torch
import clip
import torch.nn.functional as F

# CLIP 모델 로드
DEVICE = "cpu"
MODEL_NAME = "ViT-B/32"
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)

def recommend_region(choice: str, keyword: str) -> str:
    with open("place_features.json", "r", encoding='utf-8') as fp:
        place_features = json.load(fp)
    # 키워드로 부터 특성 벡터 추출
    text = clip.tokenize([keyword]).to(DEVICE)
    text_features = model.encode_text(text)
    region_similarities = {}
    # 각 지역의 특성과 비교하여 유사도 계산 (코사인 유사도 사용)
    for feature in place_features:
        region_name = feature['region'] # 각 항목에서 지역 이름 접근
        region_features = torch.tensor(feature['features']).unsqueeze(0).to(DEVICE)
        similarity = F.cosine_similarity(text_features, region_features).mean().item()
        region_similarities[region_name] = similarity

    # 유사도 정렬하여 관련성 높은 지역 추천
    recommended_region = sorted(region_similarities.items(), key=lambda x: x[1], reverse=True)[0][0]
    print(f"사용자 유형 '{choice}의 ': 추천지역은 '{recommended_region}'.")
    return recommended_region

def recommend_places_in_region(region: str) -> list:
    with open("place_features.json", "r", encoding='utf-8') as fp:
        place_features = json.load(fp)
    place_similarities = {}
    for feature in place_features:
        if feature['region'] == region:
            place_name = feature['place']
            similarity = torch.nn.functional.pairwise_distance(torch.tensor(feature['features']), torch.tensor([0]*len(feature['features']))).sum().item()
            place_similarities[place_name] = similarity
    recommended_places = sorted(place_similarities.items(), key=lambda x: x[1])[:5]
    print(f"지역에 따른 '{region}': 추천여행지는 {[place for place, _ in recommended_places]}.")
    return recommended_places

def recommend_music(place_name: str) -> list:
    with open("place_features.json", "r", encoding='utf-8') as fp:
        place_features = json.load(fp)
    with open("music_features.json", "r", encoding='utf-8') as fp:
        music_features = json.load(fp)
    place_feature = next((place['features'] for place in place_features if place['place'] == place_name), None)
    similarities = {}
    for music_name, details in music_features.items():
        music_feature = torch.tensor(details['embedding'])
        similarity = torch.nn.functional.pairwise_distance(torch.tensor(place_feature), music_feature).sum().item()
        similarities[music_name] = similarity
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1])[:5]
    recommended_music = [music for music, _ in sorted_similarities]
    print(f"장소 '{place_name}': 추천 음악 {recommended_music}.")
    return recommended_music

def process_all_recommendations(data):
    new_results = []
    for item in data:
        recommand_choice = item['Recommand_choice']
        recommand_keyword = item['Recommand_keyword']
        # 함수 호출 시 두 개의 필요한 인자 모두 전달
        recommended_region = recommend_region(recommand_choice, recommand_keyword)
        recommended_places = recommend_places_in_region(recommended_region)
        place_music_map = {}
        for place, _ in recommended_places:
            music_list = recommend_music(place)
            place_music_map[place] = music_list
        new_results.append({
            "Recommand_choice": recommand_choice,
            "Recommand_keyword": recommand_keyword,
            "Recommended_region": recommended_region,
            "Place_to_music_map": place_music_map
        })
    with open('melodymap_results2.json', 'w', encoding='utf-8') as file:
        json.dump(new_results, file, ensure_ascii=False, indent=2)
    print("All recommendations have been processed and saved.")
    return new_results

# 데이터 불러오기
with open('melodymap_modify.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 전체 추천 과정 실행
final_results = process_all_recommendations(data)
