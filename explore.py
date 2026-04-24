import pandas as pd

df = pd.read_csv('train.tsv', sep='\t')
print("Shape:", df.shape)
print("\nNull counts:")
print(df.isnull().sum())
print("\nPrice stats:")
print(df['price'].describe())
print("\nSample:")
print(df.head(3))
