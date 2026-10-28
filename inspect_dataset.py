import pandas as pd
import json

df = pd.read_csv("data/interim/hiv_future_virologic_prediction.csv")

print("DATASET SHAPE:", df.shape)

print("\nCOLUMN TYPES AND MISSING VALUES:")
summary = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "non_missing": df.notna().sum()
})
print(summary.to_string())

for col in ["lab_records", "medication_records"]:
    print(f"\n--- {col} ---")
    values = df[col].dropna()
    print("Non-empty rows:", len(values))

    if len(values) == 0:
        continue

    value = values.iloc[0]

    try:
        obj = json.loads(value) if isinstance(value, str) else value
        print("Record type:", type(obj).__name__)

        if isinstance(obj, dict):
            print("Available keys:", list(obj.keys()))
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
            print("Available record keys:", list(obj[0].keys()))
        else:
            print("Structure preview:", str(obj)[:200])
    except (json.JSONDecodeError, TypeError):
        print("Not JSON-formatted.")
        print("Stored value type:", type(value).__name__)
