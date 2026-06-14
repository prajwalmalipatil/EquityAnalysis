import pandas as pd

df = pd.DataFrame({
    'Date': ['2026-06-10', '2026-06-11', '2026-06-12'],
    'Volume': [100, 200, 300],
    'High': [10, 11, 12],
    'Low': [9, 10, 11]
})
df['str_date'] = df['Date'].astype(str).str.split().str[0]
seq_trigger_date = '2026-06-12'

eval_df = df[df['str_date'] > seq_trigger_date].reset_index(drop=True)
print("eval_df length:", len(eval_df))
