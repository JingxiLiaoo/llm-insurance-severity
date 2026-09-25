from pathlib import Path
from collections import Counter
import re
import pandas as pd

data_folder = Path(".") 
output = Path("output")
output.mkdir(exist_ok=True)

# Read and clean the descriptions.
tr = pd.read_csv(data_folder / "peril.training.csv")
te = pd.read_csv(data_folder / "peril.validation.csv")
for df in (tr, te):
    df["Description"] = df["Description"].str.strip()
combined = pd.concat([tr, te], ignore_index=True)

# Summarize the training, validation, and combined datasets.
summary = pd.DataFrame([
    [name, years, len(df), df["Description"].nunique(),
     df["Loss"].mean(), df["Loss"].median()]
    for name, years, df in [
        ("Training", "2006–2010", tr),
        ("Validation", "2011", te),
        ("Combined", "2006–2011", combined)]
], columns=["Set", "Claim years", "Records", "Unique descriptions",
            "Mean loss", "Median loss"])

# Extract unique descriptions and sort alphabetically.
texts = combined["Description"].dropna().drop_duplicates().sort_values()
texts.to_frame().to_csv(output / "unique_descriptions.csv",
                       index=False, encoding="utf-8-sig")

# Lowercase for lexical statistics; exclude numeric and mixed tokens.
tokens = texts.str.lower().str.findall(r"\b\w+(?:'\w+)?\b")
words = [[w for w in row if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", w)]
         for row in tokens]
freq = Counter(w for row in words for w in row)
n_words, n_types, n_texts = sum(freq.values()), len(freq), len(texts)

# Count adjacent word pairs without bridging excluded tokens.
bigrams = {(a, b) for row in tokens for a, b in zip(row, row[1:])
           if a in freq and b in freq}

lexical = pd.DataFrame([
    ["Number of unique descriptions", n_texts],
    ["Total number of words", n_words],
    ["Number of distinct words", n_types],
    ["Mean words per description", n_words / n_texts],
    ["Number of distinct adjacent word pairs", len(bigrams)],
    ["Distinct words occurring only once (%)",
     100 * sum(n == 1 for n in freq.values()) / n_types],
    ["Descriptions containing low-frequency words (%)",
     100 * sum(any(freq[w] <= 2 for w in row) for row in words) / n_texts]
], columns=["Statistic", "Value"])

# Save the two tables.
for name, table in [("data_summary_table", summary),
                    ("lexical_statistics", lexical)]:
    table.to_csv(output / f"{name}.csv", index=False,
                 encoding="utf-8-sig", float_format="%.2f")
    print(table.to_string(index=False, float_format="%.2f"))