import json
from collections import Counter

# JSON 파일 로드
with open('010melodymap_results.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 추천된 지역 추출
recommended_regions = [item['places'] for item in data]

# 각 지역별 추천 횟수 계산
region_counts = Counter(recommended_regions)

# 결과 출력
for region, count in region_counts.items():
    print(f"{region}: {count}개")

#############################################
# 추천된 여행지 목록 추출 및 개수 계산
place_counts = {}

# # 모든 데이터 항목에 대해 반복
# for entry in data:
#     # 각 항목의 여행지 추천 목록을 가져와 place_counts에 추가
#     for region, places in entry['Place_to_music_map'].items():
#         if region in place_counts:
#             place_counts[region] += len(places)
#         else:
#             place_counts[region] = len(places)
#
# # 결과 출력
# for region, count in place_counts.items():
#     print(f"{region}{count}개")

##############################################
# 모든 지역의 음악 리스트 추출 및 통합

# all_music = []
# for entry in data:
#         for music_list in entry['Place_to_music_map'].values():
#                 all_music.extend(music_list)
#
# # 음악별 추천 횟수 계산
# music_counts = Counter(all_music)
#
# # 결과 출력
# for music, count in music_counts.items():
#     print(f"음악 '{music}' {count}개.")