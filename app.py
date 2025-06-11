import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Modeli eğitme
df = pd.read_csv('dataset/heart.csv')
y = df['output']
x = df.drop('output', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=16)

tree = DecisionTreeClassifier()
model = tree.fit(x_train, y_train)


st.title("Kalp Krizi Riski Tahmini")

age = st.slider("Yaş", 20, 80, 50)
sex = st.selectbox("Cinsiyet (0: Kadın, 1: Erkek)", [0, 1])
cp = st.selectbox("Göğüs Ağrısı Tipi (0-3)", [0, 1, 2, 3])
trestbps = st.slider("Dinlenme Kan Basıncı", 80, 200, 120)
chol = st.slider("Kolesterol", 100, 400, 200)
fbs = st.selectbox("Açlık Kan Şekeri > 120 mg/dl (1: Evet, 0: Hayır)", [0, 1])
restecg = st.selectbox("Dinlenme EKG Sonucu (0-2)", [0, 1, 2])
thalach = st.slider("Maksimum Kalp Atış Hızı", 60, 220, 150)
exang = st.selectbox("Egzersize Bağlı Anjina (1: Evet, 0: Hayır)", [0, 1])
oldpeak = st.slider("ST Depresyonu", 0.0, 6.0, 1.0)
slope = st.selectbox("Eğim (0-2)", [0, 1, 2])
ca = st.selectbox("Damar Sayısı (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal (0-3)", [0, 1, 2, 3])

if st.button("Tahmin Et"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("Dikkat! Kalp krizi riski var!")
    else:
        st.success("Kalp krizi riski düşük.")
