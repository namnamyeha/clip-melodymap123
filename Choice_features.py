import json
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

def extract_features_from_data():
    """ CSV파일에서 데이터를 읽어 임베딩을 추출합니다. """
    df = pd.read_json('melodymap_modify.json')

    features_list = []

    for Recommand_choice,Recommand_keyword in tqdm(zip(df['Recommand_choice'],df['Recommand_keyword']), desc="임베딩 중", total=df.shape[0]):
        try:
            response = client.embeddings.create(
                input=[Recommand_keyword],
                model="text-embedding-3-small",
                dimentions = 512
            )
            embedding = response.data[0].embedding
            features_list.append({
                "choice": Recommand_choice,
                "embedding": embedding
                })
        except Exception as e:
            print(f"{['Recommand_choice']} 처리 중 오류 발생: {e}")
            continue

    # 추출된 특징을 JSON 파일로 저장
    with open("extracted_choice_features.json", "w", encoding='utf-8') as fp:
        json.dump(features_list, fp, ensure_ascii=False, indent=4)

# 함수 호출
extract_features_from_data()