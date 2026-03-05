import pandas as pd

df = pd.read_csv(r"data\\mix_train\\large_portfolio.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.head(500).to_csv(r"data\\mix_train\\portfolio.csv", index=False)