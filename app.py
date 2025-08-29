
import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM
from data import load_data

# --- Configuración Streamlit ---
st.write("# Chat with Credit Card Fraud Dataset 🦙")

# Carga de datos
df = load_data("./data")

with st.expander("🔎 Dataframe Preview"):
    st.write(df.tail(3))

# Campo de texto para la consulta
query = st.text_area("🗣️ Chat with Dataframe")

if query:
    # LLM a través de LiteLLM (Gemini en este caso)
    llm = LiteLLM(
        model="gemini/gemini-2.5-flash",
        api_key=st.secrets["gemini_key"],
        temperature=0.5
    )

    # SmartDataframe con el LLM configurado
    sdf = SmartDataframe(df, config={"llm": llm})

    # Ejecuta la consulta
    response = sdf.chat(query)

    # Manejo de la respuesta (igual que en el ejemplo sin callback)
    if response.type == "dataframe":
        st.dataframe(response.value, use_container_width=True, hide_index=True)
        st.code(response.last_code_executed, language="python")

    elif response.type == "chart":
        st.image(response.value)
        st.code(response.last_code_executed, language="python")

    else:
        st.write(response.value)
        st.code(response.last_code_executed, language="python")
