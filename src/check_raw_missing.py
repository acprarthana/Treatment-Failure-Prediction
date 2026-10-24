
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "data" / "lab_rslt.dat"

# SAS positions are 1-based and inclusive.
# VLOAD = columns 195-202; LEU3N = columns 116-119.
fields = {
    "LEU3N": (116, 119),
    "VLOAD": (195, 202),
    "R2SVL": (211, 217),
    "RTQ2VL": (219, 225),
}

counts = {
    name: {"blank": 0, "period": 0, "other": 0}
    for name in fields
}

examples = {name: [] for name in fields}

with RAW_FILE.open("r", encoding="ascii", errors="replace") as file:
    for line_number, line in enumerate(file, start=1):
        for name, (start, end) in fields.items():
            value = line[start - 1:end].strip()

            if value == "":
                counts[name]["blank"] += 1
            elif value == ".":
                counts[name]["period"] += 1
                if len(examples[name]) < 3:
                    examples[name].append((line_number, repr(value)))
            else:
                counts[name]["other"] += 1

print("Raw file:", RAW_FILE)
print("\nRaw fixed-width field counts:")
for name in fields:
    print(f"\n{name}: {counts[name]}")
    print("Period examples:", examples[name])

print("\nDone. Original data was not modified.")
