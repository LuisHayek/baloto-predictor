import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelo
modelo = joblib.load("modelo_loteria_xgboost.pkl")

st.set_page_config(page_title="Predicción Baloto", page_icon="🎰")
st.title("🔮 Predicción de Números Baloto")
st.write("""
Esta app genera combinaciones de números y predice cuáles tienen mayor probabilidad de ser ganadoras, usando un modelo XGBoost entrenado con datos históricos.
""")

# Botón para generar combinaciones
if st.button("🎲 Generar y predecir 10 combinaciones"):
    combinaciones = []
    for _ in range(1000):
        nums = sorted(np.random.choice(range(1, 44), size=5, replace=False))
        especial = np.random.choice(range(1, 17), size=1)
        combinaciones.append(nums + list(especial))

    df = pd.DataFrame(combinaciones, columns=["A", "B", "C", "D", "E", "F"])
    df["suma_grupo_principal"] = df[["A", "B", "C", "D", "E"]].sum(axis=1)
    df["num_especial"] = df["F"]
    df["probabilidad_ganador"] = modelo.predict_proba(df[["suma_grupo_principal", "num_especial"]])[:, 1]

    top_10 = df.sort_values(by="probabilidad_ganador", ascending=False).head(10)

    st.subheader("🔝 Top 10 combinaciones más probables:")
    st.dataframe(top_10[["A", "B", "C", "D", "E", "F", "probabilidad_ganador"]])
