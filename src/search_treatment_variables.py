
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent

FILES = [
    ROOT / "data" / "raw" / "sasinp" / "lab_rslt.inp",
    ROOT / "data" / "raw" / "codebook" / "lab_rslt.cdb",
    ROOT / "data" / "raw" / "sasinp" / "outcome.inp",
    ROOT / "data" / "raw" / "codebook" / "outcome.cdb",
]

KEYWORDS = [
    "antiretroviral",
    "antiretroviral therapy",
    "treatment",
    "therapy",
    "haart",
    "art status",
    "drug",
    "regimen",
    "viral load",
    "virologic failure",
    "virological failure",
    "suppression",
    "rebound",
    "adherence",
    "medication",
]

def main():
    for path in FILES:
        print("\n" + "=" * 80)
        print("FILE:", path.name)

        if not path.exists():
            print("NOT FOUND:", path)
            continue

        lines = path.read_text(
            encoding="latin-1",
            errors="replace"
        ).splitlines()

        matches = []
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in KEYWORDS):
                matches.append((i + 1, line.strip()))

        if not matches:
            print("No keyword matches.")
            continue

        print("Matching lines:", len(matches))
        for line_no, line in matches[:100]:
            print(f"{line_no}: {line[:250]}")

        if len(matches) > 100:
            print("Output limited to first 100 matches.")

if __name__ == "__main__":
    main()
