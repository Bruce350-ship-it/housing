import os
import sys
import pandas as pd
from pathlib import Path

# Ensure project root is on sys.path when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import basic_cleaning


def test_basic_cleaning_creates_output(tmp_path: Path):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "processed.csv"

    df = pd.DataFrame({
        "price": [100, 100],
        "area": [2000, 2000],
        "bedrooms": [3, 3],
        "bathrooms": [2, 2],
        "stories": [2, 2],
        "parking": [1, 1],
        "mainroad": [" Yes ", " Yes "],
        "guestroom": ["No", "No"]
    })
    df.to_csv(raw, index=False)

    cleaned = basic_cleaning(str(raw), str(out))
    assert out.exists()
    assert cleaned["mainroad"].iloc[0] == "yes"
    assert cleaned["guestroom"].iloc[0] == "no"


