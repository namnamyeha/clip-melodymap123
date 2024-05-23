import json
import torch
from numpy.random import choice

def merge(reommendations):
  merged_data = []

  with open("Travle/Travel_pre.json", "r", encoding='utf-8') as fp:
    travel_data = json.load(fp)
    for user_type, recommendation in reommendations.items():
      for place in recommendation:
        for region in travel_data:
          if place in region["places"]:
            merged_data.append(region["places"][place])
  return merged_data

def get_recommendation() ->  dict[str, list[str]]:
  with open("place_image_features.json", "r", encoding='utf-8') as fp:
    place_features = json.load(fp)
    unique_places = set()
    with open("extracted_choice_features.json", "r", encoding='utf-8') as fp:
      user_preference_features = json.load(fp)

      similarities = {}
      for user_type, user_preference in user_preference_features.items():
        similarities[user_type] = {}
        for place in place_features:
          name = place["region"]
          place_name = place["place"]
          feature = place["features"]

          # similarity as euclidean distance
          similarity = torch.nn.functional.pairwise_distance(
            torch.tensor(user_preference),
            torch.tensor(feature[0])
          )
          # similarity by cosine similarity
          similarities[user_type][place_name] = similarity.item()

        # perform weighted sampling using similarities
        similarity_sum = sum([similarity * similarity for similarity in
                              similarities[user_type].values()])
        similarity_weights = [similarity * similarity / similarity_sum for
                              similarity in similarities[user_type].values()]
        similarities[user_type] = choice(list(similarities[user_type].keys()), 5,
                                         p=similarity_weights)
        print(similarities)
  return similarities




