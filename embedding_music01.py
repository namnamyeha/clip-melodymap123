# place embedding

from collections import Counter
import numpy as np
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


# json 파일 로드
music_data = pd.read_json('Music_pre.json')

# 모든 관련 속성을 하나의 문자열로 결합
music_data['info'] = music_data.apply(lambda row: ' '.join([
    str(row['music_genre']),
    str(row['music_title']),
    str(row['music_singer']),
    str(row['music_lyric'])
]), axis=1)

# 임베딩과 관련된 정보를 저장할 리스트
embeddings_list = []

for music_title, music_singer, info in tqdm(music_data[['music_title', 'music_singer', 'info']].values, desc="임베딩 중"):
    print(f"처리 중: {music_title}")
    words = info.split()
    word_freq = Counter(words)
    top_words = [word for word, freq in word_freq.most_common(10)]
    print(f"상위 10개 단어: {top_words}")

    word_embeddings = []
    for word in top_words:
        response = client.embeddings.create(
            input=word,
            model="text-embedding-3-small",
            dimensions=1536
        )
        try:
            embedding = response.data[0].embedding
            word_embeddings.append(embedding)
            print(f"{word}의 임베딩 완료")
        except KeyError:
            print(f"임베딩 실패: {word}")

    if word_embeddings:
        avg_embedding = np.mean(word_embeddings, axis=0)
        embeddings_list.append({
            "music_title": music_title,
            "music_singer": music_singer,
            "embedding": avg_embedding.tolist()
        })
        print(f"{music_title}의 평균 임베딩 계산 완료")

# JSON 파일로 저장
with open('Data/Music-embedding01.json', 'w', encoding='utf-8') as file:
    json.dump(embeddings_list, file, ensure_ascii=False, indent=4)
    print("JSON 파일 저장 완료")
