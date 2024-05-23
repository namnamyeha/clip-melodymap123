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

# CSV 파일 로드
travel_data = pd.read_csv('Travle/TravleData - TravleData.csv')

# 모든 관련 속성을 하나의 문자열로 결합
travel_data['info'] = travel_data.apply(lambda row: ' '.join([
    str(row['poi_region']),
    str(row['poi_name']),
    ' '.join(eval(row['poi_tag'])),  # 태그는 리스트로 저장되어 있다고 가정 (eval 사용)
    str(row['img_rname']),
    str(row['poi_desc'])
]), axis=1)

# 임베딩과 관련된 정보를 저장할 리스트
embeddings_list = []

for poi_name, info in tqdm(zip(travel_data['poi_name'],travel_data['info']),desc="임베딩 중"):
    response =client.embeddings.create(
        input = info,
        model="text-embedding-3-small"
    )
    try :
       embedding = response.data[0].embedding
       embeddings_list.append((poi_name,embedding))
    except KeyError:
        print("임베딩 실패")

embeddings_df = pd.DataFrame(embeddings_list,  columns=['poi_name', 'embedding'])

with open("place_embedding.json", "w", encoding='utf-8') as fp:
    json.dump(embeddings_list,fp, ensure_ascii=False, indent=4)


