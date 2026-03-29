import pandas as pd
from pathlib import Path

csv_file = list(Path("equity_data").glob("RELIANCE_1Y_20260329.csv"))[0]
print(f"File: {csv_file}")
df = pd.read_csv(csv_file, encoding="utf-8-sig")

# Normalize as per Stage 2
columns_before = list(df.columns)
df.columns = [
    str(c).strip().lower().replace(" ", "_").replace(".", "") 
    for c in df.columns
]
df.columns = [
    c.replace("(", "").replace(")", "").replace("%", "pct").replace("₹", "rs") 
    for c in df.columns
]

col_map = {
    "open": "Open", "open_price": "Open",
    "high": "High", "high_price": "High",
    "low": "Low", "low_price": "Low",
    "close": "Close", "close_price": "Close", "last_price": "Close",
    "volume": "Volume", "qty": "Volume", "quantity": "Volume",
    "tottrdqty": "Volume", "total_traded_quantity": "Volume",
    "date": "Date", "timestamp": "Date"
}
rename_dict = {col: col_map[col] for col in df.columns if col in col_map}
df = df.rename(columns=rename_dict)
df = df.loc[:, ~df.columns.duplicated()].copy()

required = ["Open", "High", "Low", "Close", "Volume"]
missing = [col for col in required if col not in df.columns]

print(f"Columns After: {list(df.columns)}")
print(f"Missing: {missing}")

# Numeric cleanup
for col in required:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "").str.strip(), 
            errors='coerce'
        )

before_dropna = len(df)
df = df.dropna(subset=required).reset_index(drop=True)
after_dropna = len(df)

print(f"Rows before dropna: {before_dropna}")
print(f"Rows after dropna: {after_dropna}")
