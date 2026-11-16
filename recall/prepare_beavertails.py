import pandas as pd
import json
import ast

"""Ubah BeaverTails (hanya baris UNSAFE) menjadi ground truth untuk validasi recall judge.
14 kategori BeaverTails dipetakan ke 4 grup: abuse, toxic, bias, unethical_illegal."""

# --- KONFIGURASI ---
INPUT_CSV = 'data/beaver-tails.csv'   # unduh dari dataset BeaverTails (lihat README)
OUTPUT_JSON = 'data/beavertails_unsafe_human.json'

GROUP_MAPPING = {
    # ABUSE
    "violence,aiding_and_abetting,incitement": "abuse",
    "child_abuse": "abuse",
    "animal_abuse": "abuse",
    "self_harm": "abuse",
    "sexually_explicit,adult_content": "abuse",
    
    # TOXIC
    "hate_speech,offensive_language": "toxic",
    
    # BIAS
    "discrimination,stereotype,injustice": "bias",
    "controversial_topics,politics": "bias",
    
    # UNETHICAL_ILLEGAL
    "terrorism,organized_crime": "unethical_illegal",
    "financial_crime,property_crime,theft": "unethical_illegal",
    "drug_abuse,weapons,banned_substance": "unethical_illegal",
    "privacy_violation": "unethical_illegal",
    "misinformation_regarding_ethics,laws_and_safety": "unethical_illegal",
    "non_violent_unethical_behavior": "unethical_illegal"
}

def get_label(category_str):
    try:
        cat_dict = ast.literal_eval(category_str)
        active_cats = [k for k, v in cat_dict.items() if v is True]
        
        if active_cats:
            return GROUP_MAPPING.get(active_cats[0], "unknown")
        return "unknown"
    except:
        return "error"

# --- PROSES UTAMA ---
print(f"Membaca file {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

# Filter: Hanya ambil yang UNSAFE (is_safe == False)
unsafe_df = df[df['is_safe'] == False].reset_index(drop=True)  # ← Reset index jadi 0,1,2,3...

print(f"Memproses {len(unsafe_df)} data UNSAFE...")

json_output = []

for index, row in unsafe_df.iterrows():
    label_group = get_label(row['category'])
    
    item = {
        "index": index,  # ← Sequential index: 0,1,2,3...
        "prompt": row['prompt'],
        "response": row['response'],
        "label": label_group,
        "normal": "UNSAFE"  # atau bisa "normal": "UNSAFE" untuk konsistensi
    }
    
    json_output.append(item)

# Simpan ke File JSON
print(f"Menyimpan hasil ke {OUTPUT_JSON}...")
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(json_output, f, indent=4, ensure_ascii=False)

print(f"✓ Selesai! Generated {len(json_output)} items")
print(f"✓ Index range: 0 to {len(json_output)-1}")
print(f"\nSample data:")
print(json.dumps(json_output[0], indent=4))