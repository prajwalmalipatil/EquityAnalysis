import pandas as pd
from pathlib import Path
import os

results_path = Path('equity_data/Results')
confirmed, failed, pending = 0, 0, 0
total_processed = 0

print(f"Auditing {len(list(results_path.glob('*.xlsx')))} files...")

for f in results_path.glob('*.xlsx'):
    try:
        df = pd.read_excel(f, sheet_name='VSA_Analysis')
        if 'Validation_Status' not in df.columns:
            continue
            
        stats = df['Validation_Status'].value_counts()
        confirmed += stats.get('Confirmed ✅', 0)
        failed += stats.get('Failed ❌', 0)
        pending += stats.get('Pending ⏳', 0)
        total_processed += 1
    except Exception as e:
        continue

total = confirmed + failed
win_rate = (confirmed / total * 100) if total > 0 else 0

print("\n" + "="*40)
print(f"--- GLOBAL WIN-RATE AUDIT ---")
print(f"SYMBOLS AUDITED:  {total_processed}")
print(f"TOTAL SIGNALS:    {total}")
print(f"CONFIRMED ✅:     {confirmed}")
print(f"FAILED ❌:        {failed}")
print(f"PENDING ⏳:       {pending}")
print(f"-"*40)
print(f"WIN RATE:         {win_rate:.2f}%")
print("="*40 + "\n")
