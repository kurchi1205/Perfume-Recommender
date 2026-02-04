import pandas as pd
import json
import re

def extract_notes_from_description(description):
    """Extract top, middle, and base notes from description text"""
    notes = {
        "top": [],
        "middle": [],
        "base": []
    }
    
    if pd.isna(description) or not str(description).strip():
        return notes
    
    desc_str = str(description)
    
    # Pattern to match: "Top notes are X, Y and Z; middle notes are A, B and C; base notes are D, E and F"
    # Also handles variations like "Top note is X" or "Top notes: X, Y"
    
    # Extract top notes
    top_pattern = r'[Tt]op notes?\s+(?:are|is|:)\s+([^;\.]+?)(?:;|\.|\smiddle)'
    top_match = re.search(top_pattern, desc_str)
    if top_match:
        top_notes_str = top_match.group(1)
        notes["top"] = parse_notes_from_text(top_notes_str)
    
    # Extract middle notes
    middle_pattern = r'[Mm]iddle notes?\s+(?:are|is|:)\s+([^;\.]+?)(?:;|\.|\sbase)'
    middle_match = re.search(middle_pattern, desc_str)
    if middle_match:
        middle_notes_str = middle_match.group(1)
        notes["middle"] = parse_notes_from_text(middle_notes_str)
    
    # Extract base notes
    base_pattern = r'[Bb]ase notes?\s+(?:are|is|:)\s+([^;\.]+?)(?:\.|$)'
    base_match = re.search(base_pattern, desc_str)
    if base_match:
        base_notes_str = base_match.group(1)
        notes["base"] = parse_notes_from_text(base_notes_str)
    
    return notes

def parse_notes_from_text(notes_text):
    """Parse notes from natural language text (handles 'and', commas, etc.)"""
    if not notes_text:
        return []
    
    # Replace ' and ' with ', ' for uniform splitting
    notes_text = re.sub(r'\s+and\s+', ', ', notes_text)
    
    # Split by comma and clean
    notes = [note.strip() for note in notes_text.split(',') if note.strip()]
    
    return notes

def parse_notes(notes_string):
    """Parse comma-separated notes string into a list"""
    if pd.isna(notes_string) or str(notes_string).strip() == '' or str(notes_string).lower() == 'unknown':
        return []
    return [note.strip() for note in str(notes_string).split(',') if note.strip()]

def parse_main_accords(*accord_sources):
    """Parse main accords from multiple sources and combine them"""
    accords = []
    
    for source in accord_sources:
        if pd.isna(source) or not str(source).strip():
            continue
            
        source_str = str(source).strip()
        
        # If it's a string representation of a list
        if source_str.startswith('[') and source_str.endswith(']'):
            try:
                parsed = eval(source_str)
                accords.extend(parsed)
            except:
                pass
        # If it's a regular string, add it
        elif source_str and source_str.lower() != 'none':
            accords.append(source_str)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_accords = []
    for accord in accords:
        if accord and accord not in seen:
            seen.add(accord)
            unique_accords.append(accord)
    
    return unique_accords

def extract_brand_from_name(name):
    """Extract brand from name field (last word before 'for')"""
    if pd.isna(name):
        return None
    
    name_str = str(name).strip()
    match = re.search(r'(.+?)\s+(for\s+(?:women|men|women\s+and\s+men|unisex))', name_str)
    if match:
        before_gender = match.group(1).strip()
        parts = before_gender.split()
        if len(parts) > 1:
            return parts[-1]
    return None

def row_to_perfume_json(row):
    """Convert a merged dataframe row to the desired JSON structure"""
    
    # Check if this record is only in CSV2
    is_csv2_only = pd.isna(row.get('Perfume'))
    
    # Name: prefer Perfume from CSV1, fallback to Name from CSV2
    name = None
    if pd.notna(row.get('Perfume')):
        name = str(row['Perfume']).strip()
    elif pd.notna(row.get('Name')):
        # Clean the name from CSV2
        name_str = str(row['Name']).strip()
        name = re.sub(r'for\s+(?:women|men|women\s+and\s+men|unisex)', '', name_str).strip()
        # Also remove the brand if present
        brand_in_name = extract_brand_from_name(row.get('Name'))
        if brand_in_name:
            name = name.replace(brand_in_name, '').strip()
    
    # Brand: from CSV1, or extract from Name in CSV2, can be null
    brand = None
    if pd.notna(row.get('Brand')):
        brand = str(row['Brand']).strip()
    elif is_csv2_only and pd.notna(row.get('Name')):
        brand = extract_brand_from_name(row.get('Name'))
    
    # Gender: prefer CSV1, fallback to CSV2
    gender = None
    if pd.notna(row.get('Gender_x')):  # Gender from CSV1
        gender = str(row['Gender_x']).strip()
    elif pd.notna(row.get('Gender_y')):  # Gender from CSV2
        gender = str(row['Gender_y']).strip()
    
    # Description: from CSV2
    description = str(row['Description']).strip() if pd.notna(row.get('Description')) else None
    
    # Notes: from CSV1, or extract from description if CSV2 only
    if not is_csv2_only:
        notes = {
            "top": parse_notes(row.get('Top')),
            "middle": parse_notes(row.get('Middle')),
            "base": parse_notes(row.get('Base'))
        }
    else:
        # Extract notes from description for CSV2-only records
        notes = extract_notes_from_description(description)
    
    # Main accords: combine from both sources
    main_accords = parse_main_accords(
        row.get('Main Accords'),
        row.get('mainaccord1'),
        row.get('mainaccord2'),
        row.get('mainaccord3'),
        row.get('mainaccord4'),
        row.get('mainaccord5')
    )
    if len(main_accords) == 0:
        return {}
    
    # Year: from CSV1
    year_released = None
    if pd.notna(row.get('Year')):
        year_str = row['Year']
        year_released = int(year_str)
    
    # URL
    url = str(row['url']).strip() if pd.notna(row.get('url')) else None
    
    return {
        "name": name,
        "brand": brand,
        "gender": gender,
        "description": description,
        "notes": notes,
        "main_accords": main_accords,
        "year_released": year_released,
        "url": url
    }

# Main execution
print("Reading CSV files...")
print("-" * 80)

# Read CSV1 (semicolon-separated)
df1 = pd.read_csv('../../datasets/fragrantica/fra_cleaned.csv', sep=';', encoding='latin-1')
print(f"CSV1 shape: {df1.shape}")
print(f"CSV1 columns: {list(df1.columns)}")

# Read CSV2 (comma-separated)
df2 = pd.read_csv('../../datasets/fragrantica/fra_perfumes.csv', encoding='latin-1')
print(f"\nCSV2 shape: {df2.shape}")
print(f"CSV2 columns: {list(df2.columns)}")

# Check for URL column
if 'url' not in df1.columns or 'url' not in df2.columns:
    print("\nError: 'url' column not found in one or both CSVs")
    exit(1)

# Perform outer join on URL to keep all records from both CSVs
print("\nPerforming outer join on 'url' column...")
df_merged = pd.merge(df1, df2, on='url', how='outer', suffixes=('_x', '_y'))

print(f"Merged dataframe shape: {df_merged.shape}")

# Statistics
only_csv1 = df_merged[df_merged['Name'].isna()]
only_csv2 = df_merged[df_merged['Perfume'].isna()]
both = df_merged[df_merged['Name'].notna() & df_merged['Perfume'].notna()]

print(f"\nJoin Statistics:")
print(f"  - Records in both CSVs: {len(both)}")
print(f"  - Records only in CSV1: {len(only_csv1)}")
print(f"  - Records only in CSV2: {len(only_csv2)}")
print(f"  - Total unique records: {len(df_merged)}")

# Convert to JSON
print("\nConverting to JSON format...")
print("-" * 80)

perfumes = []
for idx, row in df_merged.iterrows():
    perfume = row_to_perfume_json(row)
    if perfume == {}:
        continue
    perfumes.append(perfume)

# Save to JSON
output_file = 'perfumes_merged.json'
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(perfumes, json_file, indent=2, ensure_ascii=False)

print(f"✓ Created {output_file} with {len(perfumes)} perfumes")

# Display sample outputs
print("\n" + "=" * 80)
print("Sample perfume (from both CSVs):")
print("=" * 80)
if len(both) > 0:
    sample_idx = both.index[0]
    print(json.dumps(perfumes[sample_idx], indent=2, ensure_ascii=False))

if len(only_csv1) > 0:
    print("\n" + "=" * 80)
    print("Sample perfume (CSV1 only):")
    print("=" * 80)
    sample_idx = only_csv1.index[0]
    print(json.dumps(perfumes[sample_idx], indent=2, ensure_ascii=False))

if len(only_csv2) > 0:
    print("\n" + "=" * 80)
    print("Sample perfume (CSV2 only - with extracted notes):")
    print("=" * 80)
    sample_idx = only_csv2.index[0]
    print(json.dumps(perfumes[sample_idx], indent=2, ensure_ascii=False))

# Optional: Save the merged dataframe as CSV for inspection
df_merged.to_csv('merged_dataframe.csv', index=False, encoding='utf-8')
print(f"\n✓ Also saved merged dataframe to 'merged_dataframe.csv' for inspection")

print("\n✓ Processing complete!")