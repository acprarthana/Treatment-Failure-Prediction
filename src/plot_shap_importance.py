
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
input_file = ROOT / "reports" / "shap_feature_importance.csv"
output_file = ROOT / "reports" / "shap_feature_importance.png"

df = pd.read_csv(input_file)

required_columns = {"Feature", "Mean_Absolute_SHAP"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Expected columns {required_columns}, got {df.columns.tolist()}"
    )

df = df.dropna(subset=["Feature", "Mean_Absolute_SHAP"])
df = df.sort_values("Mean_Absolute_SHAP", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(df["Feature"], df["Mean_Absolute_SHAP"])
ax.set_title("Global SHAP Feature Importance")
ax.set_xlabel("Mean Absolute SHAP Value")
ax.set_ylabel("Model Feature")
ax.grid(axis="x", alpha=0.25)

fig.tight_layout()
fig.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Chart saved to: {output_file}")
print("\nFeatures displayed:")
print(df.sort_values("Mean_Absolute_SHAP", ascending=False).to_string(index=False))
