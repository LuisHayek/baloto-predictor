import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelo
modelo = joblib.load("modelo_loteria_xgboost.pkl")

st.set_page_config(page_title="Predicción Baloto", page_icon="🎰")
st.title("🔮 Predicción de Números Baloto")
st.write("""
Esta app genera combinaciones de números y predice cuáles tienen mayor probabilidad de ser ganadoras,
usando un modelo XGBoost entrenado con datos históricos. 
         Nota: 1, 2, 3, 4, 5 → son los 5 números principales del 1 al 43 (sin repetir) y 6 → es el número especial o balota adicional del 1 al 16.

📌 App y modelo creados por **Luis Alejandro Gutierrez Hayek-Big Data Analytics**.
""")

# --- Sección: Generador de combinaciones automáticas ---
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

    # Renombrar columnas para claridad
    top_10 = top_10.rename(columns={
        'A': 'Número 1',
        'B': 'Número 2',
        'C': 'Número 3',
        'D': 'Número 4',
        'E': 'Número 5',
        'F': 'Balota Especial',
        'probabilidad_ganador': 'Probabilidad'
    })

    st.subheader("🔝 Top 10 combinaciones más probables:")
    st.dataframe(top_10[["Número 1", "Número 2", "Número 3", "Número 4", "Número 5", "Balota Especial", "Probabilidad"]])

# --- Sección: Predicción manual ---
st.markdown("---")
st.header("🎯 Verifica tu combinación manual")

col1, col2, col3, col4, col5, col6 = st.columns(6)

n1 = col1.number_input("Número 1", min_value=1, max_value=43, step=1)
n2 = col2.number_input("Número 2", min_value=1, max_value=43, step=1)
n3 = col3.number_input("Número 3", min_value=1, max_value=43, step=1)
n4 = col4.number_input("Número 4", min_value=1, max_value=43, step=1)
n5 = col5.number_input("Número 5", min_value=1, max_value=43, step=1)
balota = col6.number_input("Balota Especial", min_value=1, max_value=16, step=1)

if st.button("🔍 Predecir mi combinación"):
    numeros_principales = [n1, n2, n3, n4, n5]

    if len(set(numeros_principales)) < 5:
        st.error("❌ Los números principales no deben repetirse.")
    else:
        suma = sum(numeros_principales)
        df_pred = pd.DataFrame([[suma, balota]], columns=['suma_grupo_principal', 'num_especial'])
        proba = modelo.predict_proba(df_pred)[0][1]
        resultado = "🎉 ¡Alta probabilidad de ganar!" if proba > 0.7 else "🤔 Probabilidad baja, inténtalo con otros números"
        st.success(f"{resultado} (Probabilidad: {proba:.4f})")

