import pandas as pd

# Veri setini oku
df=pd.read_csv("data/students.csv")

# İlk beş satırı getir
print(df.head())

# Veri setinin boyutu
print(df.shape)

# veri setinin sütunları
print(df.columns)

df.info()