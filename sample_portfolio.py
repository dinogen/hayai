import pandas as pd

df = pd.read_csv("data\\mix_2\\portfolio.csv")
df = df.sample(frac=1).reset_index(drop=True)
df.head(50).to_csv("data\\mix_2\\portfolio_sample.csv", index=False)
