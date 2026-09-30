##For model that has a separate thinking mode
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from langchain_ollama import ChatOllama

# 1. Model settings
#````````````````````````
model_id = "deepseek-r1:1.5b"

output_folder = Path("deepseek-r1_1.5b_results0823")
output_folder.mkdir(parents=True, exist_ok=True)
#````````````````````````


# 2. Read datasets
#````````````````````````
tr = pd.read_csv("peril.training.csv")
te = pd.read_csv("peril.validation.csv")

tr["Description"] = tr["Description"].astype(str).str.strip()
te["Description"] = te["Description"].astype(str).str.strip()

dat2 = pd.concat([tr, te], axis=0)
descriptions = sorted(dat2["Description"].dropna().unique())

print("Number of unique descriptions:", len(descriptions))

description_to_index = {
    description: index
    for index, description in enumerate(descriptions)
}
#````````````````````````


# 3. Start runtime
#````````````````````````
checkpoint0 = datetime.now()
#````````````````````````


# 4. Configure model
#````````````````````````
ollama_server_url = "http://localhost:11434"

#If model has a separate thinking mode, we turn off the thinking mode. (The two GPT-OSS models ould only be reduced to the lowest level.)
if model_id.startswith("gpt-oss:"):
    reasoning_setting = "low"
else:
    reasoning_setting = False
llm = ChatOllama(model=model_id, base_url=ollama_server_url, num_thread=10, reasoning=reasoning_setting)


checkpoint1 = datetime.now()
#````````````````````````


# 5. Categories
#````````````````````````
categories = ["low", "high"]
#categories = ["low", "medium", "high"]
#categories = ["negligible", "low", "medium", "high", "catastrophic"]
levels = len(categories)
#````````````````````````


# 6. Instruction
#````````````````````````
instruction = (
    "You are an actuarial assistant classifying insurance incidents.\n"

    f"Classify each incident into exactly one of these labels: "
    f"{', '.join(categories)}.\n"

    "OUTPUT RULES:\n"
    "Use lowercase only.\n"
    "Use the same numbering as the input.\n"
    "Return exactly one numbered line for every input incident.\n"
    "Do not skip any item.\n"
    "Do not repeat or summarize the incident description.\n"
    "Do not explain your reasoning.\n"
    "Do not add the incident description before or after the label.\n"
    "Do not use markdown, brackets, parentheses, or additional text.\n"
    "Any output other than the exact allowed labels is invalid.\n\n"

    "The only valid output format is:\n"
    "1. <allowed label>\n"
    "2. <allowed label>\n"
    "3. <allowed label>\n"
    "...\n"
)
#````````````````````````


# 7. Run classification
#````````````````````````
N = len(descriptions)
batch_size = 20
labels = []

for beg in range(0, N, batch_size):
    end = min(N, beg + batch_size)
    current_batch_size = end - beg

    print()
    print("=" * 70)
    print(f"Processing descriptions {beg + 1} to {end} of {N}")
    print("=" * 70)

    prompt = (
        f"The following are descriptions of "
        f"{current_batch_size} insurance incidents:"
    )

    for j in range(beg, end):
        prompt += f"\n{j - beg + 1}. {descriptions[j]}"

    prompt += (
        "\n\n"
        + instruction
        + f"\nReturn exactly {current_batch_size} numbered lines."
    )

    try:
        response = llm.invoke(prompt)
        response_text = response.content

        reasoning_content = response.additional_kwargs.get(
            "reasoning_content"
        )

        if reasoning_setting is False and reasoning_content:
            print("Warning: thinking was generated but hidden.")

    except Exception as error:
        print("The LLM request failed:")
        print(error)
        labels.extend([""] * current_batch_size)
        continue

    print()
    print("LLM raw response:")
    print(response_text)
    print("-" * 70)

    batch_labels = [None] * current_batch_size

    for raw_line in response_text.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        match = re.match(r"^(\d+)\s*[\.\):\-]\s*(.*)$", line)

        if match is None:
            continue

        number = int(match.group(1))
        raw_label = match.group(2).strip()

        if 1 <= number <= current_batch_size:
            batch_labels[number - 1] = raw_label

    missing_numbers = [
        i + 1
        for i, value in enumerate(batch_labels)
        if value is None
    ]

    if missing_numbers:
        print(
            "Warning: the following items in this batch "
            "did not receive a numbered label:"
        )
        print(missing_numbers)

    batch_labels = [
        value if value is not None else ""
        for value in batch_labels
    ]

    labels.extend(batch_labels)
#````````````````````````


# 8. Runtime
#````````````````````````
checkpoint2 = datetime.now()

print()
print("=" * 70)
print("Classification completed.")
print("Number of unique descriptions:", len(descriptions))
print("Number of labels stored:", len(labels))
print("=" * 70)

if len(labels) != len(descriptions):
    raise RuntimeError(
        "The number of stored labels does not match "
        "the number of descriptions."
    )
#````````````````````````


# 9. Map labels back
#````````````````````````
tr["raw_label"] = [
    labels[description_to_index[description]]
    for description in tr["Description"]
]

te["raw_label"] = [
    labels[description_to_index[description]]
    for description in te["Description"]
]
#````````````````````````


# 10. Runtime output
#````````````````````````
total_runtime = checkpoint2 - checkpoint0
request_runtime = checkpoint2 - checkpoint1

runtime_output = pd.DataFrame({
    "model": [model_id],
    "n unique cases": [len(descriptions)],
    "n training rows": [len(tr)],
    "n validation rows": [len(te)],
    "empty unique labels": [sum(label == "" for label in labels)],
    "invalid unique labels": [sum(label not in categories and label != "" for label in labels)],
    "total runtime (min)": [total_runtime.total_seconds() / 60],
    "request runtime (min)": [request_runtime.total_seconds() / 60]
})
#````````````````````````


# 11. Save results
#````````````````````````
safe_model_name = model_id.replace(":", "_")

training_output_file = output_folder / f"{safe_model_name}_{levels}cat_training.csv"
validation_output_file = output_folder / f"{safe_model_name}_{levels}cat_validation.csv"
runtime_output_file = output_folder / f"{safe_model_name}_{levels}cat_runtime.csv"

tr.to_csv(training_output_file, index=False)
te.to_csv(validation_output_file, index=False)
runtime_output.to_csv(runtime_output_file, index=False)

print()
print(f"Training results saved to: {training_output_file}")
print(f"Validation results saved to: {validation_output_file}")
print(f"Runtime results saved to: {runtime_output_file}")
#````````````````````````








##For model that does not have a separate thinking mode
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from langchain_ollama import ChatOllama

# 1. Model settings
#````````````````````````
model_id = "llama3.1:8b"

output_folder = Path("llama3.1_8b_results0823")
output_folder.mkdir(parents=True, exist_ok=True)
#````````````````````````


# 2. Read datasets
#````````````````````````
tr = pd.read_csv("peril.training.csv")
te = pd.read_csv("peril.validation.csv")

tr["Description"] = tr["Description"].astype(str).str.strip()
te["Description"] = te["Description"].astype(str).str.strip()

dat2 = pd.concat([tr, te], axis=0)
descriptions = sorted(dat2["Description"].dropna().unique())

print("Number of unique descriptions:", len(descriptions))

description_to_index = {
    description: index
    for index, description in enumerate(descriptions)
}
#````````````````````````


# 3. Start runtime
#````````````````````````
checkpoint0 = datetime.now()
#````````````````````````


# 4. Configure model
#````````````````````````
ollama_server_url = "http://localhost:11434"

llm = ChatOllama(model=model_id, base_url=ollama_server_url, num_thread=10)


checkpoint1 = datetime.now()
#````````````````````````


# 5. Categories
#````````````````````````
categories = ["low", "high"]
#categories = ["low", "medium", "high"]
#categories = ["negligible", "low", "medium", "high", "catastrophic"]
levels = len(categories)
#````````````````````````


# 6. Instruction
#````````````````````````
instruction = (
    "You are an actuarial assistant classifying insurance incidents.\n"

    f"Classify each incident into exactly one of these labels: "
    f"{', '.join(categories)}.\n"

    "OUTPUT RULES:\n"
    "Use lowercase only.\n"
    "Use the same numbering as the input.\n"
    "Return exactly one numbered line for every input incident.\n"
    "Do not skip any item.\n"
    "Do not repeat or summarize the incident description.\n"
    "Do not explain your reasoning.\n"
    "Do not add the incident description before or after the label.\n"
    "Do not use markdown, brackets, parentheses, or additional text.\n"
    "Any output other than the exact allowed labels is invalid.\n\n"

    "The only valid output format is:\n"
    "1. <allowed label>\n"
    "2. <allowed label>\n"
    "3. <allowed label>\n"
    "...\n"
)
#````````````````````````


# 7. Run classification
#````````````````````````
N = len(descriptions)
batch_size = 20
labels = []

for beg in range(0, N, batch_size):
    end = min(N, beg + batch_size)
    current_batch_size = end - beg

    print()
    print("=" * 70)
    print(f"Processing descriptions {beg + 1} to {end} of {N}")
    print("=" * 70)

    prompt = (
        f"The following are descriptions of "
        f"{current_batch_size} insurance incidents:"
    )

    for j in range(beg, end):
        prompt += f"\n{j - beg + 1}. {descriptions[j]}"

    prompt += (
        "\n\n"
        + instruction
        + f"\nReturn exactly {current_batch_size} numbered lines."
    )

    try:
        response = llm.invoke(prompt)
        response_text = response.content

    except Exception as error:
        print("The LLM request failed:")
        print(error)
        labels.extend([""] * current_batch_size)
        continue

    print()
    print("LLM raw response:")
    print(response_text)
    print("-" * 70)

    batch_labels = [None] * current_batch_size

    for raw_line in response_text.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        match = re.match(r"^(\d+)\s*[\.\):\-]\s*(.*)$", line)

        if match is None:
            continue

        number = int(match.group(1))
        raw_label = match.group(2).strip()

        if 1 <= number <= current_batch_size:
            batch_labels[number - 1] = raw_label

    missing_numbers = [
        i + 1
        for i, value in enumerate(batch_labels)
        if value is None
    ]

    if missing_numbers:
        print(
            "Warning: the following items in this batch "
            "did not receive a numbered label:"
        )
        print(missing_numbers)

    batch_labels = [
        value if value is not None else ""
        for value in batch_labels
    ]

    labels.extend(batch_labels)
#````````````````````````


# 8. Runtime
#````````````````````````
checkpoint2 = datetime.now()

print()
print("=" * 70)
print("Classification completed.")
print("Number of unique descriptions:", len(descriptions))
print("Number of labels stored:", len(labels))
print("=" * 70)

if len(labels) != len(descriptions):
    raise RuntimeError(
        "The number of stored labels does not match "
        "the number of descriptions."
    )
#````````````````````````


# 9. Map labels back
#````````````````````````
tr["raw_label"] = [
    labels[description_to_index[description]]
    for description in tr["Description"]
]

te["raw_label"] = [
    labels[description_to_index[description]]
    for description in te["Description"]
]
#````````````````````````


# 10. Runtime output
#````````````````````````
total_runtime = checkpoint2 - checkpoint0
request_runtime = checkpoint2 - checkpoint1

runtime_output = pd.DataFrame({
    "model": [model_id],
    "n unique cases": [len(descriptions)],
    "n training rows": [len(tr)],
    "n validation rows": [len(te)],
    "empty unique labels": [sum(label == "" for label in labels)],
    "invalid unique labels": [sum(label not in categories and label != "" for label in labels)],
    "total runtime (min)": [total_runtime.total_seconds() / 60],
    "request runtime (min)": [request_runtime.total_seconds() / 60]
})
#````````````````````````


# 11. Save results
#````````````````````````
safe_model_name = model_id.replace(":", "_")

training_output_file = output_folder / f"{safe_model_name}_{levels}cat_training.csv"
validation_output_file = output_folder / f"{safe_model_name}_{levels}cat_validation.csv"
runtime_output_file = output_folder / f"{safe_model_name}_{levels}cat_runtime.csv"

tr.to_csv(training_output_file, index=False)
te.to_csv(validation_output_file, index=False)
runtime_output.to_csv(runtime_output_file, index=False)

print()
print(f"Training results saved to: {training_output_file}")
print(f"Validation results saved to: {validation_output_file}")
print(f"Runtime results saved to: {runtime_output_file}")
#````````````````````````