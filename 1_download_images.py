
import csv
import requests

with (open('Travle/TravleData - TravleData.csv', 'r', encoding='utf-8') as file):
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        # skip the header
        if row[3] == '이미지URL':
            continue
        print(row)
        url = row[3]
        response = requests.get(url)

        region = row[0]
        name = row[1]
        # replace the space with underscore
        name = region + name.replace(' ', '_')
        with open(f'images/{name}.jpg', 'wb') as file:
            file.write(response.content)