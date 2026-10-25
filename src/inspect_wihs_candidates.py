
from pathlib import Path
import pandas as pd

ROOT = Path("data/raw/WIHS/WIHS PDS/CSV")

FILES = [
    "labsum.csv", "drug1.csv", "drug2.csv",
    "drug3.csv", "hivstat.csv", "outcome.csv"
]

def read_csv_chunks(path):
    """Try common encodings without changing the source file."""
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            # Test whether the whole file can be read in chunks
            chunks = pd.read_csv(
                path, chunksize=50000,
                low_memory=False, encoding=encoding
            )
            rows = 0
            for chunk in chunks:
                rows += len(chunk)

            return rows, encoding

        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode {path.name}")

for filename in FILES:
    path = ROOT / filename
    print(f"\n{'=' * 60}\n{filename}")

    # Read only the header
    header = pd.read_csv(path, nrows=0, encoding="cp1252")
    print("Columns:", list(header.columns))

    rows, encoding = read_csv_chunks(path)
    print("Rows:", rows)
    print("Encoding used:", encoding)
