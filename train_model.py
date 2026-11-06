import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os


from sklearn.model_selection import train_test_split   #Veriyi eğitim ve test olarak ayırır.
from sklearn.linear_model import LinearRegression   #Doğrusal regresyon algoritmasını oluşturur.

from sklearn.metrics import mean_squared_error #MSE hesaplar.
from sklearn.metrics import mean_absolute_error #MAE hesaplar.
from sklearn.metrics import r2_score #R² değerini hesaplar.

# Veri setini oku
df=pd.read_csv("data/students.csv")

# İlk 5 satırı göster
print("VERİ SETİ")
print(df.head())

#Bağımsız değişken
X=df[["study_hours", "previous_score"]]

#X: Modelimize vereceğimiz bilgiler.

#y: Modelimizin tahmin etmeyi öğreneceği değer.

#Bağımlı değişken
y=df["exam_score"]

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Eğitim veri sayısı: ",len(X_train))
print("Test veri sayısı: ", len(X_test))

"""
Makine öğrenmesinde modelin yalnızca ezberlemesini istemiyoruz. Daha önce görmediği veriler üzerinde de başarılı tahminler yapmasını istiyoruz.

Bu nedenle verileri iki gruba ayırıyoruz:
- Eğitim verisi: Modelin öğrenmesi için kullanılır.
- Test verisi: Modelin daha önce görmediği örneklerdeki başarısını ölçmek için kullanılır.
test_size=0.20 demek verilerin %20'sini test için ayırmak demektir.

Bizim 20 satırlık veri setimizde:
- 16 satır eğitim.
- 4 satır test.
random_state=42 ise veri bölme işleminin tekrarlanabilir olmasını sağlar.
"""

#Modeli oluştur
model=LinearRegression()

#Modeli eğit
model.fit(X_train,y_train)

print("Model eğitimi tamamlandı")

# Test verileri üzerinde tahmin yap
y_pred=model.predict(X_test)


print("\nGERÇEK VE TAHMİN EDİLEN DEĞERLER")

for gercek, tahmin in zip(y_test,y_pred):
     print(f"Gerçek: {gercek:.2f} | Tahmin: {tahmin:.2f}")


"""
Burada:
- predict() tahmin yapar.
- y_test gerçek sınav notlarını tutar.
- y_pred modelin tahmin ettiği notları tutar.
- zip() gerçek ve tahmin değerlerini eşleştirir.
"""

#mse
mse=mean_squared_error(y_test,y_pred)

#rmse
rmse=np.sqrt(mse)

#mae
mae=mean_absolute_error(y_test,y_pred)

#r2
r2=r2_score(y_test,y_pred)

print("\nMODEL DEĞERLENDİRME SONUÇLARI")

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)

"""
Metrik	İyi sonuç nasıl olmalı?
MSE	Genellikle 0'a yakın
RMSE	Genellikle 0'a yakın
MAE	Genellikle 0'a yakın
R²	1'e yakın
"""

print("\nMODEL KATSAYILARI")

print("Intercept:", model.intercept_)

for ozellik, katsayi in zip(X.columns, model.coef_):
    print(f"{ozellik}: {katsayi}")


# GERÇEK VE TAHMİN EDİLEN DEĞERLERİN GÖRSELLEŞTİRİLMESİ
plt.figure(figsize=(8,5))

plt.scatter(y_test,y_pred)

# Mükemmel tahmin doğrusunu çiz
plt.plot(
     [y_test.min(),y_test.max()],
     [y_test.min(),y_test.max()],
     linestyle="--"
)

plt.xlabel("Gerçek sınav Notları")
plt.ylabel("Tahmin edilen sınav notları")
plt.title("Gerçek ve Tahmin Edilen Notların Karşılaştırılması")

plt.grid()
plt.show()

"""
Bu grafik bize ne anlatacak?
- X ekseni: Gerçek sınav notları.
- Y ekseni: Modelin tahmin ettiği sınav notları.
- Kesikli çizgi: Modelin hatasız tahmin yapması durumunda noktaların üzerinde bulunacağı referans doğrusu.
Noktalar bu çizgiye ne kadar yakınsa tahminlerimiz o kadar iyidir.
"""

# Modeli kaydetmek için klasör oluştur
os.makedirs("models",exist_ok=True)

metrics={
    "MSE": mse,
    "RMSE": rmse,
    "MAE": mae,
    "R2 Score": r2
}

joblib.dump(metrics, "models/model_metrics.pkl")
# Eğitilmiş modeli kaydet
joblib.dump(model, "models/student_score_model.pkl")

print("Model başarıyla kaydedildi!")

"""
Bu kodlar ne yapıyor?
- import joblib: Eğittiğimiz makine öğrenmesi modelini dosyaya kaydetmemizi sağlar.
- os.makedirs("models", exist_ok=True): models klasörü yoksa oluşturur. Varsa tekrar oluşturmaya çalışmaz.
- joblib.dump(): Eğittiğimiz model nesnesini .pkl uzantılı dosyaya kaydeder.
Önemli: Bu kodu model.fit() işleminden sonra, yani model eğitildikten sonra çalıştırmalıyız.
"""