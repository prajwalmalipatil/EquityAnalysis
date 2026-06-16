import pytest
import pandas as pd
import shutil
from pathlib import Path
from src.services.vsa.data_quality_service import DataQualityService

@pytest.fixture
def test_dir():
    base = Path("test_dq_data")
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    yield base
    if base.exists():
        shutil.rmtree(base)

def test_data_quality_service(test_dir):
    service = DataQualityService(test_dir)
    
    # 1. Valid File
    valid_df = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "Open": [100, 101, 102, 103, 104],
        "High": [105, 106, 107, 108, 109],
        "Low": [95, 96, 97, 98, 99],
        "Close": [102, 103, 104, 105, 106],
        "Volume": [1000, 1000, 1000, 1000, 1000]
    })
    valid_df.to_csv(test_dir / "valid.csv", index=False)
    
    # 2. Duplicate Date
    dup_df = valid_df.copy()
    dup_df.loc[1, "Date"] = "2023-01-01"
    dup_df.to_csv(test_dir / "dup.csv", index=False)
    
    # 3. Invalid OHLC
    inv_df = valid_df.copy()
    inv_df.loc[0, "Low"] = 110  # Low > High
    inv_df.to_csv(test_dir / "invalid_ohlc.csv", index=False)
    
    # 4. Zero Volume
    zero_df = valid_df.copy()
    zero_df["Volume"] = 0
    zero_df.to_csv(test_dir / "zero_vol.csv", index=False)

    stats = service.run_gate()
    
    assert stats["total_files"] == 4
    assert stats["passed"] == 1
    assert stats["quarantined"] == 3
    
    assert (test_dir / "valid.csv").exists()
    assert not (test_dir / "dup.csv").exists()
    assert (test_dir / "quarantine" / "dup.csv").exists()
    assert "DUPLICATE_DATES" in stats["reasons"]
    assert "INVALID_OHLC_PRICES" in stats["reasons"]
    assert "ZERO_TOTAL_VOLUME" in stats["reasons"]

def test_series_filtering(test_dir):
    service = DataQualityService(test_dir)
    
    # DataFrame with EQ and BL series on duplicate dates
    mixed_df = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "Series": ["EQ", "BL", "EQ", "W2", "EQ", "EQ", "EQ"],
        "Open": [100, 100, 101, 101, 102, 103, 104],
        "High": [105, 105, 106, 106, 107, 108, 109],
        "Low": [95, 95, 96, 96, 97, 98, 99],
        "Close": [102, 102, 103, 103, 104, 105, 106],
        "Volume": [1000, 5000, 1000, 2000, 1000, 1000, 1000]
    })
    mixed_df.to_csv(test_dir / "mixed.csv", index=False)
    
    stats = service.run_gate()
    assert stats["total_files"] == 1
    assert stats["passed"] == 1
    assert stats["quarantined"] == 0
    assert (test_dir / "mixed.csv").exists()
    
    # Read the cleaned file and check that only EQ rows remain (meaning no duplicates)
    cleaned_df = pd.read_csv(test_dir / "mixed.csv")
    assert len(cleaned_df) == 5
    assert not cleaned_df['Date'].duplicated().any()
    assert (cleaned_df['Series'] == 'EQ').all()

