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

# music_genre,music_title,music_singer,music_lyric,music_image
def extract_features_from_data():
    """ CSV파일에서 데이터를 읽어 임베딩을 추출합니다. """
    df = pd.read_csv('Music_info.csv')

    # 모든 관련 속성을 하나의 문자열로 결합
    df['info'] =df.apply(lambda row: ' '.join([
        str(row['music_genre']),
        str(row['music_title']),
        str(row['music_singer']),
        str(row['music_lyric']),
        str(row['music_image']),

    ]), axis=1)

    features_list = []

    for music_title,music_singer, info in tqdm(zip(df['music_title'],df['music_singer'],df['info']), desc="임베딩 중", total=df.shape[0]):
        try:
            response = client.embeddings.create(
                input=[info],
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            features_list.append({
                "music": music_title,
                "singer": music_singer,
                "embedding": embedding
                })
        except Exception as e:
            print(f"{['music']} 처리 중 오류 발생: {e}")
            continue

    # 추출된 특징을 JSON 파일로 저장
    with open("e_music_features.json", "w", encoding='utf-8') as fp:
        json.dump(features_list, fp, ensure_ascii=False, indent=4)

# 함수 호출
extract_features_from_data()