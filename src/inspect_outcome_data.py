
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "raw" / "data" / "outcome.dat"
LAYOUT_FILE = ROOT / "data" / "raw" / "sasinp" / "outcome.inp"

# Read the SAS input layout as text.
layout_text = LAYOUT_FILE.read_text(
    encoding="utf-8",
    errors="replace"
)

print("=" * 65)
print("OUTCOME FILE INSPECTION")
print("=" * 65)

print("\n1. Raw file information")
print("File exists:", DATA_FILE.exists())
print("File size (bytes):", DATA_FILE.stat().st_size)

# Inspect raw record lengths before attempting to parse fields.
lengths = {}

with DATA_FILE.open("r", encoding="ascii", errors="replace") as f:
    for line in f:
        length = len(line.rstrip("\r\n"))
        lengths[length] = lengths.get(length, 0) + 1

print("Record lengths and counts:", lengths)

print("\n2. Important field definitions from SAS layout")

lines = layout_text.splitlines()

keywords = [
    "CASEID",
    "AIDSCASE",
    "TAIDS",
    "DETERM",
    "DATE1mm",
    "DATE1yy",
    "AIDSREPD",
    "DEATH",
    "DTHDATE",
    "DTHREPDT",
]

for i, line in enumerate(lines):
    if any(word.lower() in line.lower() for word in keywords):
        print(line.rstrip())

print("\n3. Raw sample records (first 3 lines)")

with DATA_FILE.open("r", encoding="ascii", errors="replace") as f:
    for i, line in enumerate(f):
        if i == 3:
            break
        print(f"Record {i + 1}: {line.rstrip()}")

print("\nInspection only: no files were changed.")

