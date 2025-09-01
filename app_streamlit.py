import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_mlbb.joblib")

st.title("MLBB Classification")
kill = st.slider("Jumlah Kill",0,20)
assist = st.slider("Jumlah Assist",0,20)
death = st.slider("Jumlah Death",0,20)
turret = st.slider("Jumlah Turret",0,20)

if st.button("Predict"):
    data_baru = pd.DataFrame([[kill,assist,death,turret]],columns=["kill","assist","death","turret"])
    st.success(f"Hasil prediksi : {model.predict(data_baru)[0]}")
    st.balloons()