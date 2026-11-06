import streamlit as st
import streamlit as st
import joblib
import pandas as pd

model=joblib.load("models/student_score_model.pkl")
metrics=joblib.load("models/model_metrics.pkl")


st.title("Öğrenci sınav notu tahmin uygulaması")

st.write(
     "Bu uygulama, öğrencinin çalışma saati ve önceki sınav notunu kullanarak tahmini sınav notunu hesaplar."
)

st.divider()

st.subheader("📝 Öğrenci Bilgileri")
st.caption("Bilgilerinizi girerek sınav notunuzu tahmin edebilirsiniz.")

# Kullanıcıdan çalışma saati bilgisi al
study_hours=st.number_input(
    "Günde kaç saat çalışıyorsunuz?",
    min_value=0.0,
    max_value=24.0,
    value=None,
    step=0.5
)

# Kullanıcıdan önceki sınav notunu al
previous_score =st.number_input(
    "Önceki sınav notunuz kaç?",
    min_value=0.0,
    max_value=100.0,
    value=None,
    step=1.0
)

# Tahmin butonu
if st.button("Sınav Notumu Tahmin Et"):
        # Kullanıcı verilerini DataFrame'e dönüştür
        input_data= pd.DataFrame({
                "study_hours":[study_hours],
                "previous_score":[previous_score]
        })

        # Model ile tahmin yap
        prediction=model.predict(input_data)
      
        # Tahmini 0-100 aralığında tut
        prediction = prediction[0]
        prediction = max(0, min(100, prediction))

        # Tahmin sonucunu ekranda göster
        st.success("Tahmin işlemi başarıyla tamamlandı!")

        st.metric(
            label="Tahmini Sınav Notunuz",
            value=f"{prediction:.2f} / 100"
        )

        # Not göstergesi
        st.progress(int(prediction))

st.divider()
st.subheader("📊 Model Performansı")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MSE", f"{metrics['MSE']:.2f}")
col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
col3.metric("MAE", f"{metrics['MAE']:.2f}")
col4.metric("R² Score", f"{metrics['R2 Score']:.4f}")

