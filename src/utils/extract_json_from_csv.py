import csv
import json
import ast
from collections import defaultdict

# ---------------------------
# Helpers
# ---------------------------

PATH_TO_DATA_1 = "../../datasets/fragrantica/fra_perfumes.csv"
PATH_TO_DATA_2 = "../../datasets/fragrantica/fra_cleaned.csv"

def parse_list_field(value):
    """Safely parse stringified lists"""
    if not value or value.strip() == "":
        return []
    try:
        return ast.literal_eval(value)
    except Exception:
        return []

def split_notes(note_str):
    if not note_str:
        return []
    return [n.strip() for n in note_str.split(",") if n.strip()]

# ---------------------------
# Load CSV 1 (Fragrantica dump)
# ---------------------------

csv1_by_url = {}

with open(PATH_TO_DATA_1, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        url = row["url"].strip()
        csv1_by_url[url] = {
            "name": row["Name"].split("by")[0].strip(),
            "brand": row["Name"].split("by")[-1].strip() if "by" in row["Name"] else None,
            "gender": row["Gender"].strip(),
            "description": row["Description"].strip(),
            "main_accords": parse_list_field(row["Main Accords"]),
        }


# ---------------------------
# Load CSV 2 (Structured notes)
# ---------------------------

csv2_by_url = {}

with open(PATH_TO_DATA_2, newline="", encoding="latin-1") as f:
    reader = csv.DictReader(f, delimiter=";")
    for row in reader:
        url = row["url"].strip()
        csv2_by_url[url] = {
            "name": row["Perfume"].strip(),
            "brand": row["Brand"].strip(),
            "gender": row["Gender"].strip(),
            "year_released": int(row["Year"]) if row["Year"].isdigit() else None,
            "notes": {
                "top": split_notes(row["Top"]),
                "middle": split_notes(row["Middle"]),
                "base": split_notes(row["Base"]),
            },
            "main_accords": [
                row.get("mainaccord1"),
                row.get("mainaccord2"),
                row.get("mainaccord3"),
                row.get("mainaccord4"),
                row.get("mainaccord5"),
            ],
        }

# ---------------------------
# Merge
# ---------------------------

all_urls = set(csv2_by_url)
final_data = []

for url in all_urls:
    c1 = csv1_by_url.get(url, {})
    c2 = csv2_by_url.get(url, {})
    if not c2.get("brand"):
        print(c2)
    perfume = {
        "name": c1.get("name") or c2.get("name"),
        "brand": c2.get("brand") or c1.get("brand"),
        "gender": c2.get("gender") or c1.get("gender"),
        "description": c1.get("description"),
        "notes": c2.get("notes", {"top": [], "middle": [], "base": []}),
        "main_accords": list(
            dict.fromkeys(  # de-duplicate while preserving order
                (c1.get("main_accords") or [])
                + [a for a in c2.get("main_accords", []) if a]
            )
        ),
        "image_url": None,
        "year_released": c2.get("year_released"),
        "url": url,
    }

    final_data.append(perfume)

# ---------------------------
# Write JSON
# ---------------------------

with open("../../datasets/fragrantica_perfumes.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print(f"✅ Exported {len(final_data)} perfumes to perfumes.json")
