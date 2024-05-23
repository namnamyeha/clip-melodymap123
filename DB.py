import json
import boto3

# AWS SDK 초기화 (boto3)
dynamodb = boto3.resource('dynamodb')

# DynamoDB 테이블 지정
table = dynamodb.Table('MelodyMap_result')

# JSON 파일을 열어 데이터 로드
with open('FINAL.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

    # DynamoDB 테이블에 데이터 항목 삽입
    for item in data:
        try:
            # DynamoDB에 데이터 삽입
            table.put_item(Item=item)
            print(f"Successfully inserted item: {item['result_choice']}")
        except Exception as e:
            print(f"Failed to insert item: {item['result_choice']}. Error: {str(e)}")

print("All items have been attempted to be inserted into DynamoDB.")
