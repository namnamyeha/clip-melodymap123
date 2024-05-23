import json
from itertools import chain

# JSON 파일 로드
with open("place_image_features.json", "r", encoding='utf-8') as file:
    image_data = json.load(file)

with open("place_text_features.json", "r", encoding='utf-8') as file:
    text_data = json.load(file)

combined_features = []
for img_feature, txt_feature in zip(image_data, text_data):
    if img_feature['region'] == txt_feature['poi_region'] and img_feature['place'] == txt_feature['poi_name']:
        # 이미지 특성 평탄화: 2차원 리스트를 1차원 리스트로 변환
        flattened_image_features = list(chain.from_iterable(img_feature['features']))
        # 텍스트 특성과 결합
        combined_feature = flattened_image_features + txt_feature['embedding']
        combined_features.append({
            'region': txt_feature['poi_region'],
            'place': txt_feature['poi_name'],
            'embedding': combined_feature
        })

# 통합된 특성을 JSON 파일로 저장
with open('combined_features.json', 'w', encoding='utf-8') as f:
    json.dump(combined_features, f, ensure_ascii=False, indent=4)
