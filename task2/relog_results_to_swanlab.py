import pandas as pd
from pathlib import Path
import swanlab

RUN_DIR = Path(r"runs\detect\CV_Midterm_Runs\finetune_20_epochs")
CSV_PATH = RUN_DIR / "results.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"找不到 results.csv: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

df.columns = df.columns.str.strip()

swanlab.init(
    project="CV-Midterm-PJ",
    experiment_name="YOLOv8_training_curves_from_results_csv",
    description="Re-visualize YOLOv8 training curves from existing results.csv without retraining.",
    config={
        "model": "YOLOv8",
        "task": "vehicle detection and tracking",
        "tracker": "ByteTrack",
        "source": "Ultralytics YOLOv8 results.csv",
        "note": "This visualization is generated from the original training log without retraining."
    }
)

metrics_to_log = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "lr/pg0",
    "lr/pg1",
    "lr/pg2",
]

available_columns = df.columns.tolist()
print("Available columns:")
for c in available_columns:
    print(c)

for _, row in df.iterrows():
    epoch = int(row["epoch"]) if "epoch" in df.columns else int(_)

    log_data = {}

    for metric in metrics_to_log:
        if metric in df.columns:
            value = row[metric]
            if pd.notna(value):
                log_data[metric] = float(value)

    swanlab.log(log_data, step=epoch)

swanlab.finish()

print("Done. Open SwanLab and take screenshots of the generated curves.")