import pandas as pd
df = pd.read_parquet("d100project/data/cleaned_data.parquet")
print(df.head())
print(df.shape)
