from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import os
from typing import List, Dict

app = FastAPI(title="Predict Data Simulator")

# try common filenames (the repo contains "Real estate.csv" and your tree mentions "real_estate.csv")
DATA_PATHS = [
    os.path.join(os.path.dirname(__file__), 'data', 'Real estate.csv'),
    os.path.join(os.path.dirname(__file__), 'data', 'real_estate.csv'),
]


def _find_dataset() -> str | None:
    for p in DATA_PATHS:
        if os.path.exists(p):
            return p
    return None


def _find_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    lc = [c.lower() for c in df.columns]
    for cand in candidates:
        for i, col in enumerate(lc):
            if cand in col:
                return df.columns[i]
    return None


@app.get("/fetch-data")
def fetch_data(n: int = 50) -> Dict[str, List[Dict]]:
    """Return `n` random samples mapped to fields: size, bedrooms, age, price.

    - X2 house age -> age
    - X4 number of convenience stores -> bedrooms
    - Y house price of unit area -> price
    - size: generated as random integers between 500 and 5000
    """
    path = _find_dataset()
    if not path:
        raise HTTPException(status_code=404, detail="Dataset not found in predict/data/")

    df = pd.read_csv(path)

    age_col = _find_column(df, ["x2", "house age", "x2 house age"])  # X2
    bedrooms_col = _find_column(df, ["x4", "convenience store", "number of convenience", "x4 number"])  # X4
    price_col = _find_column(df, ["y", "price", "house price", "unit area"])  # Y

    if not all([age_col, bedrooms_col, price_col]):
        raise HTTPException(
            status_code=400,
            detail=f"Required columns not found. Detected columns: {list(df.columns)}",
        )

    sample = df.sample(n=min(n, len(df))).reset_index(drop=True)
    # generate a realistic 'size' column (integers between 500 and 5000)
    sample['size'] = np.random.randint(500, 5001, size=len(sample))

    out = []
    for _, row in sample.iterrows():
        try:
            out.append({
                "size": int(row['size']),
                "bedrooms": int(row[bedrooms_col]),
                "age": float(row[age_col]),
                "price": float(row[price_col]),
            })
        except Exception:
            # skip rows that cannot be coerced
            continue

    return {"data": out}
