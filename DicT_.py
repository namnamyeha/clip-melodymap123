import json

# 추천 결과를 파일에서 로드
with open("000melodymap_results.json", "r", encoding='utf-8') as fp:
    recommendation_results = json.load(fp)

# music 리스트를 딕셔너리 형태로 변환하는 함수
def convert_music_to_dict(recommendation_results):
    for result in recommendation_results:
        for place in result["places"]:
            new_music_list = []
            for song in place["music"]:
                new_music_list.append({"title": song})  # 각 곡을 딕셔너리 형태로 변환
            place["music"] = new_music_list
    return recommendation_results

# 변환 작업 수행
updated_recommendation_results = convert_music_to_dict(recommendation_results)

# 결과를 새로운 JSON 파일에 저장
output_file = "updated_melodymap_results.json"
with open(output_file, "w", encoding='utf-8') as fp:
    json.dump(updated_recommendation_results, fp, ensure_ascii=False, indent=4)



