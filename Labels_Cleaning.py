#Label cleaning code template for the 17 models besides mistral:7b, using deepseek-r1:1.5b as an example.     
# ````````````````````````````````````````````````````````
from pathlib import Path
import re
import pandas as pd

input_folder = Path("deepseek-r1_1.5b_results0823")
output_folder = input_folder / "deepseek-r1_1.5b_cleaned_results"
output_folder.mkdir(parents=True, exist_ok=True)

file_settings = {
    "deepseek-r1_1.5b_2cat_training.csv": {"low", "high"},
    "deepseek-r1_1.5b_2cat_validation.csv": {"low", "high"},
    "deepseek-r1_1.5b_3cat_training.csv": {"low", "medium", "high"},
    "deepseek-r1_1.5b_3cat_validation.csv": {"low", "medium", "high"},
    "deepseek-r1_1.5b_5cat_training.csv": {"negligible", "low", "medium", "high", "catastrophic"},
    "deepseek-r1_1.5b_5cat_validation.csv": {"negligible", "low", "medium", "high", "catastrophic"}
}

def extract_valid_label(raw_value, valid_labels):
    if pd.isna(raw_value):
        return "NA"

    text = str(raw_value).strip().lower()
    text = re.sub(r"[*_`]+", "", text)
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "NA"

    simple = text.strip(" \t\r\n.;,!?")
    if simple in valid_labels:
        return simple

    all_labels = r"negligible|low|medium|high|catastrophic"

    # Reject ambiguous answers such as "medium or high"
    if re.search(rf"\b(?:{all_labels})\b\s*(?:or|and|to|/|&|~)\s*\b(?:{all_labels})\b", text):
        return "NA"

    labels = "|".join(sorted((re.escape(x) for x in valid_labels), key=len, reverse=True))

    patterns = [
        rf"^.*<\s*({labels})(?:\s*[- ]?\s*risk)?\s*>[.!]?$",
        rf"^<\s*allowed label\s*>\s*({labels})(?:\s*[- ]?\s*risk)?[.!]?$",
        rf"^(?:allowed\s+label|label|classification(?:\s+category)?|category|risk\s+level)\s*[:=\-]\s*({labels})(?:\s*[- ]?\s*risk)?[.!]?$",
        rf"^\d+\s*[\.\):\-]\s*({labels})(?:\s*[- ]?\s*risk)?[.!]?$",
        rf"^({labels})\s*[- ]?\s*(?:risk|severity|damage|impact)[.!]?$",
        rf"^({labels})\s+(?:risk|severity|damage|impact)\b.*$",
        rf"^.*?[:\-→,]\s*({labels})(?:\s*[- ]?\s*risk)?[.!]?$",
        rf"^.*?[:\-→,]\s*({labels})\s+risk\b(?:\s+(?:as|from|due to|because|which)\b.*)?[.!]?$",
        rf"^({labels})(?:\s+risk)?\s*[:\-→,]\s*.+$"
    ]

    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            return match.group(1)

    return "NA"


for filename, valid_labels in file_settings.items():
    input_path = input_folder / filename

    if not input_path.exists():
        print("File not found:", filename)
        continue

    df = pd.read_csv(input_path)

    if "raw_label" not in df.columns:
        raise KeyError(f"{filename}: raw_label column not found.")

    df["label"] = df["raw_label"].apply(lambda x: extract_valid_label(x, valid_labels))

    output_path = output_folder / filename.replace(".csv", "_clean.csv")
    df.to_csv(output_path, index=False)

    print("=" * 70)
    print(filename)
    print("Total rows:", len(df))
    print("NA:", (df["label"] == "NA").sum())
    print(df["label"].value_counts(dropna=False))
    print()


category_settings = {
    "2cat": ["deepseek-r1_1.5b_2cat_training.csv", "deepseek-r1_1.5b_2cat_validation.csv"],
    "3cat": ["deepseek-r1_1.5b_3cat_training.csv", "deepseek-r1_1.5b_3cat_validation.csv"],
    "5cat": ["deepseek-r1_1.5b_5cat_training.csv", "deepseek-r1_1.5b_5cat_validation.csv"]
}

comparison_file = output_folder / "deepseek-r1_1.5b_label_comparison.xlsx"

with pd.ExcelWriter(comparison_file, engine="openpyxl") as writer:
    for sheet_name, filenames in category_settings.items():
        parts = []

        for filename in filenames:
            raw = pd.read_csv(input_folder / filename)
            clean = pd.read_csv(output_folder / filename.replace(".csv", "_clean.csv"), keep_default_na=False)

            parts.append(pd.DataFrame({
                "Description": raw["Description"].astype("string").str.strip(),
                "raw_label": raw["raw_label"],
                "clean_label": clean["label"]
            }))

        comparison = pd.concat(parts, ignore_index=True).drop_duplicates("Description").sort_values("Description")
        comparison.to_excel(writer, sheet_name=sheet_name, index=False)

print("=" * 70)
print("Cleaning completed.")
print("Cleaned files saved to:", output_folder)
print("Comparison file saved to:", comparison_file)
#````````````````````````````````````````````````````````




#Label Cleaning code for mistral:7b.  
#````````````````````````````````````````````````````````
from pathlib import Path
import re
import pandas as pd

input_folder = Path("mistral_7b_results0823")
output_folder = input_folder / "mistral_7b_cleaned_results"
output_folder.mkdir(parents=True, exist_ok=True)

file_settings = {
    "mistral_7b_2cat_training.csv": {"low", "high"},
    "mistral_7b_2cat_validation.csv": {"low", "high"},
    "mistral_7b_3cat_training.csv": {"low", "medium", "high"},
    "mistral_7b_3cat_validation.csv": {"low", "medium", "high"},
    "mistral_7b_5cat_training.csv": {"negligible", "low", "medium", "high", "catastrophic"},
    "mistral_7b_5cat_validation.csv": {"negligible", "low", "medium", "high", "catastrophic"}
}

all_labels = ("catastrophic", "negligible", "medium", "high", "low")
label_re = "|".join(all_labels)

def extract_valid_label(raw_value, valid_labels):
    if pd.isna(raw_value): return "NA"

    text = re.sub(r"[*_`]+", "", str(raw_value).strip().lower())
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n.;,!?")

    if not text or re.fullmatch(r"high(?:[- ]school|school)", text): return "NA"

    patterns = [
        rf"^(?:\d+\s*[.):-]\s*)?<\s*({label_re})(?:\s+risk)?\s*>$",
        rf"^<\s*allowed\s+label\s*>\s*({label_re})\b",
        rf"^(?:allowed\s+label|label|classification(?:\s+category)?|category|risk\s+level)\s*[:=-]\s*({label_re})\b",
        rf"^\d+\s*[.):-]\s*({label_re})\b",
        rf"^({label_re})\b",
        rf"(?:^|[-:→,])\s*({label_re})\s*(?:[-:→,]|$)"
    ]

    match = next((m for p in patterns if (m := re.search(p, text))), None)
    if match is None: return "NA"

    label = match.group(1)
    if label not in valid_labels: return "NA"

    tail = text[match.end():]
    tail = re.sub(r"\bhigh(?:[- ]school|school)\b", "", tail)
    tail = re.sub(rf"\bnot\s+(?:an?\s+)?(?:{label_re})\b", "", tail)
    tail = re.sub(rf"\b(?:a|an|the|specific)\s+(?:{label_re})-risk\b", "", tail)

    other_labels = "|".join(x for x in all_labels if x != label)
    if re.search(rf"\b(?:{other_labels})\b", tail): return "NA"

    return label

for filename, valid_labels in file_settings.items():
    input_path = input_folder / filename
    if not input_path.exists():
        print("File not found:", filename)
        continue

    df = pd.read_csv(input_path)
    if "raw_label" not in df.columns: raise KeyError(f"{filename}: raw_label column not found.")

    df["label"] = df["raw_label"].apply(lambda x: extract_valid_label(x, valid_labels))
    output_path = output_folder / filename.replace(".csv", "_clean.csv")
    df.to_csv(output_path, index=False)

    print("=" * 70)
    print(filename)
    print("Total rows:", len(df))
    print("NA:", (df["label"] == "NA").sum())
    print(df["label"].value_counts(dropna=False))
    print()

category_settings = {
    "2cat": ["mistral_7b_2cat_training.csv", "mistral_7b_2cat_validation.csv"],
    "3cat": ["mistral_7b_3cat_training.csv", "mistral_7b_3cat_validation.csv"],
    "5cat": ["mistral_7b_5cat_training.csv", "mistral_7b_5cat_validation.csv"]
}

comparison_file = output_folder / "mistral_7b_label_comparison.xlsx"

with pd.ExcelWriter(comparison_file, engine="openpyxl") as writer:
    for sheet_name, filenames in category_settings.items():
        parts = []
        for filename in filenames:
            raw = pd.read_csv(input_folder / filename)
            clean = pd.read_csv(output_folder / filename.replace(".csv", "_clean.csv"), keep_default_na=False)
            parts.append(pd.DataFrame({
                "Description": raw["Description"].astype("string").str.strip(),
                "raw_label": raw["raw_label"],
                "clean_label": clean["label"]
            }))

        comparison = pd.concat(parts, ignore_index=True).drop_duplicates("Description").sort_values("Description")
        comparison.to_excel(writer, sheet_name=sheet_name, index=False)

print("=" * 70)
print("Cleaning completed.")
print("Cleaned files saved to:", output_folder)
print("Comparison file saved to:", comparison_file)
#````````````````````````````````````````````````````````