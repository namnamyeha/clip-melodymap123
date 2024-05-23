import json

# JSON 파일 경로
file_path = 'place_features.json'

# JSON 파일 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# data가 리스트인 경우
if isinstance(data, list):
    for item in data:
        if "features" in item:
            # 2차원 리스트를 1차원으로 평탄화
            item["features"] = [feature for sublist in item["features"] for feature in sublist]

    # JSON 파일 다시 저장
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print("JSON 파일이 성공적으로 업데이트되었습니다.")
else:
    print("data가 리스트가 아닙니다. JSON 파일의 구조를 확인하세요.")
