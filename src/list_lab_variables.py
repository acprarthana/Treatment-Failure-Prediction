
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
LAYOUT = ROOT / "data" / "raw" / "sasinp" / "lab_rslt.inp"
CODEBOOK = ROOT / "data" / "raw" / "codebook" / "lab_rslt.cdb"

# Extract variable names from the SAS INPUT layout.
text = LAYOUT.read_text(encoding="latin-1", errors="replace")
variables = re.findall(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+(?:\$?\d+(?:-\d+)?|\$\d+-\d+)",
    text,
    flags=re.MULTILINE,
)

print("LAB VARIABLES")
print("Total extracted:", len(variables))
for i, name in enumerate(variables, start=1):
    print(f"{i:3}. {name}")

print("\n" + "=" * 70)
print("CODEBOOK VARIABLE DEFINITIONS")
print("=" * 70)

book = CODEBOOK.read_text(encoding="latin-1", errors="replace").splitlines()
for i, line in enumerate(book):
    # Print lines that appear to start a variable definition.
    if re.match(r"^\s*\d+[-–]\d+\s+[A-Za-z][A-Za-z0-9_]*\s+", line):
        print(line.strip())
