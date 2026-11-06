import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv("data/students.csv")

print("Eksik değer kontrolü:")
print(df.isnull().sum())

print("\nOrtalama:")
print(df.mean())

print("\nMedyan:")
print(df.median())

print("\nStandart sapma:")
print(df.std())

print("\nKorelasyon matrisi:")
print(df.corr())

# 4. VERİ GÖRSELLEŞTİRME

# Çalışma saati ve sınav notu ilişkisi
plt.figure(figsize=(8,5))
plt.scatter(
    df["study_hours"],
    df["exam_score"]
)
plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")

plt.grid()
plt.show()

# Önceki sınav notu ve yeni sınav notu ilişkisi

plt.figure(figsize=(8,5))
plt.scatter(
    df["previous_score"],
    df["exam_score"]
)
plt.title("Previous Score vs Exam Score")
plt.xlabel("Previous Score")
plt.ylabel("Exam Score")

plt.grid()
plt.show()