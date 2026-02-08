import json

def get_unique_gender_types(data):
    unique_gender_types = set()
    for perfume in data:
        unique_gender_types.add(perfume["gender"])
    return list(unique_gender_types)



def normalize_gender_types(data):
    for perfume in data:
        if perfume["gender"] == "for men":
            perfume["gender"] = "men"
        elif perfume["gender"] == "for women":
            perfume["gender"] = "women"
        elif perfume["gender"] == "for women and men":
            perfume["gender"] = "unisex"
    return data

if __name__ == "__main__":
    JSON_PATH = "../../datasets/fragrantica_perfumes.json"

    with open(JSON_PATH, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    unique_gender_types = get_unique_gender_types(data)
    print(unique_gender_types)

    normalized_data = normalize_gender_types(data)
    unique_gender_types = get_unique_gender_types(normalized_data)
    print(unique_gender_types)
    with open(JSON_PATH, "w", encoding="utf-8") as json_file:
        json.dump(normalized_data, json_file, indent=2, ensure_ascii=False)