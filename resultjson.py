from itertools import product
import json

# 추천 여행지 조합 데이터 만들기
# 키값에 들어있는 키워드로 임베딩 하기
# 각 선택지에 대한 설명
descriptions = {
    'I': '내성적',
    'E': '적극적',
    'S': '감각적인',
    'N': '직관적',
    'T': '생각하다',
    'F': '감정적이다',
    'J': '계획',
    'P': '유연하다',
    'Z': '바다',
    'M': '산',
    'O': '자연',
    'D': '음식',
    'A': '교통수단',
    'W': '운동',
    'L': '신나는',
    'Q': '평화로운',
    'B': '당일여행',
    'X': '장기여행',
    'R': '비싸다',
    'G': '저렴하다'
}

# 각 선택지 (키 수정 반영)
choices = ['IE','SN','TF','JP','ZM','OD','AW','LQ','BX','RG']


# 모든 가능한 조합 생성
combinations = [''.join(comb) for comb in product(*choices)]

# 설명 생성
descriptions_list = []
for combination in combinations:
    description = [descriptions[choice] for choice in combination]
    descriptions_list.append(f"{combination} : {', '.join(description)}")

data = descriptions_list

new_data = []
for entry in data:
    key, value = entry.split(":")
    new_entry = {
        "Recommand_choice": key.strip(),
        "Recommand_keyword": value.strip()
    }
    new_data.append(new_entry)

# JSON 파일로 저장
with open('melodymap_modify.json', 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=2)


# import json
#
# # 원본 파일 경로
# input_file_path = 'melodymap_result.json'
#
# # 새 파일 경로
# output_file_path = 'converted_melodymap_result.json'
#
# # 원본 JSON 파일 읽기
# with open(input_file_path, 'r', encoding='cp949') as file:  # 인코딩을 cp949로 지정
#     original_data = json.load(file)
#
# # 새로운 형식으로 데이터 변환
# converted_data = []
# for item in original_data:
#     converted_item = {
#         'Recommand_choice': {'S': item['Recommand_choice']['S']},
#         'Rocommand_result': {'S': item['Rocommand_result']['S']}
#     }
#     converted_data.append(converted_item)
#
# # 변환된 데이터를 새 JSON 파일에 저장
# with open(output_file_path, 'w', encoding='utf-8') as file:  # 출력 파일은 utf-8로 인코딩
#     json.dump(converted_data, file, ensure_ascii=False, indent=2)
#
# print(f"Converted data has been saved to {output_file_path}")
