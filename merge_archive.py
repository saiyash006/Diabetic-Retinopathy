import pandas as pd
df = pd.read_csv("data/raw/Messidor/messidor_all.csv")

print(df["Id"].value_counts())