
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

comparison_file = REPORTS / "treatment_proxy_7_model_comparison.csv"
threshold_file = REPORTS / "treatment_proxy_threshold_results.csv"

comparison = pd.read_csv(comparison_file)
thresholds = pd.read_csv(threshold_file)

print("\n=== SEVEN-MODEL COMPARISON ===")
print(comparison.to_string(index=False))

print("\n=== THRESHOLD-TUNED COMPARISON ===")
print(thresholds.to_string(index=False))

# Plot seven-model ranking metrics.
metrics = ["ROC_AUC", "PR_AUC"]
available = [m for m in metrics if m in comparison.columns]
if available:
    ax = comparison.set_index("Model")[available].plot(
        kind="bar", figsize=(11, 6)
    )
    ax.set_title("Treatment Outcome Proxy: Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(REPORTS / "treatment_proxy_model_comparison.png", dpi=160)
    plt.close()

# Threshold-tuned metrics plot.
plot_cols = [
    "Test_Sensitivity", "Test_Specificity", "Test_F1"
]
plot_cols = [c for c in plot_cols if c in thresholds.columns]
if plot_cols:
    ax = thresholds.set_index("Model")[plot_cols].plot(
        kind="bar", figsize=(10, 6)
    )
    ax.set_title("Threshold-Tuned Model Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(REPORTS / "treatment_proxy_threshold_metrics.png", dpi=160)
    plt.close()

# Write a concise Markdown report.
best = thresholds.sort_values("Test_PR_AUC", ascending=False).iloc[0]
report = f"""# HIV Treatment Outcome Proxy — Model Evaluation

## Dataset and task
- Labelled rows: 2,377 (from the current training run)
- Positive labels: 262
- Negative labels: 2,115
- Input features: 10 candidate features, including derived previous-visit features
- Evaluation split: participant-level holdout
- Target: medication-stop-reason proxy, not independently confirmed clinical failure

## Threshold-tuned results
{thresholds.to_markdown(index=False)}

## Provisional result
Highest test PR-AUC among the three threshold-tuned models:
**{best['Model']}** (PR-AUC {best['Test_PR_AUC']:.3f}).

This is a preliminary result from one holdout split. Thresholds were selected on validation data.

## Limitations
1. The target is a proxy label, not confirmed clinical treatment failure.
2. Visit ordering must be validated before interpreting previous-visit features.
3. Confirm all predictors are available before the event being predicted.
4. Report precision, sensitivity, specificity, F1 and PR-AUC; accuracy alone is insufficient.
5. Validate with repeated/grouped cross-validation before claiming generalizability.
"""

(REPORTS / "treatment_proxy_final_report.md").write_text(
    report, encoding="utf-8"
)

print("\nCreated:")
print(REPORTS / "treatment_proxy_final_report.md")
print(REPORTS / "treatment_proxy_model_comparison.png")
print(REPORTS / "treatment_proxy_threshold_metrics.png")
