import json

# 임베딩 벡터 크기 확인 함수
def check_embedding_size(file_path, key):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            embedding = item[key]
            print(f"File: {file_path}, Embedding size: {len(embedding)}")
            break  # 첫 번째 항목만 확인

# 파일 경로 및 키 설정
choice_file_path = 'extracted_choice_features.json'
choice_key = 'embedding'  # extracted_choice_features.json 파일에서 임베딩 벡터가 있는 키

place_file_path = 'place_image_features.json'
place_key = 'embedding'  # place_text_features.json 파일에서 피처 벡터가 있는 키

# 임베딩 벡터 크기 확인
check_embedding_size(choice_file_path, choice_key)
check_embedding_size(place_file_path, place_key)

