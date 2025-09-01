import os
import streamlit as st
import pandas as pd
import pandasai as pai
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM
from data import load_data
import io
from PIL import Image

# --- Configuración Streamlit y Título ---
st.set_page_config(layout="wide")
st.write("# Chat with Credit Card Fraud Dataset 🦙")

# --- CLAVE #1: CONFIGURACIÓN GLOBAL DE LA MEMORIA ---
# Establecemos las reglas del juego para CUALQUIER instancia de PandasAI que se cree.
# Hacemos esto una sola vez, al principio del script.
llm = LiteLLM(
    model="gemini/gemini-2.5-flash",
    api_key=st.secrets["gemini_key"],
    temperature=0.5
)
pai.config.set({
    "llm": llm,
    'history_size': 10, # La IA recordará las 5 últimas preguntas y respuestas.
    "enable_cache": False
})

# --- Carga de datos ---
df = load_data("./data")

# --- CLAVE #2: CREACIÓN DE UN AGENTE PERSISTENTE ---
# Creamos nuestro "analista" una sola vez y lo guardamos en la sesión para que no se reinicie.
if "agent" not in st.session_state:
    st.session_state.agent = SmartDataframe(df)

with st.expander("🔎 Dataframe Preview"):
    st.write(df.tail(3))

# --- Gestión del Historial Visual ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Lógica del Chat ---
if prompt := st.chat_input("🗣️ Chat with Dataframe"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Usamos el AGENTE PERSISTENTE de la sesión. Este es el que tiene la memoria.
            response = st.session_state.agent.chat(prompt)

            response_content = response.value
            st.write(response_content)

            # (Aquí iría la lógica para mostrar gráficos o tablas si la respuesta los tuviera)
            if response.type == "dataframe":
                st.dataframe(response_content, use_container_width=True, hide_index=True)
            elif response.type == "chart":
                st.image(response_content)
            else:
                st.write(response_content)

        # Mostramos el código ejecutado en un desplegable para no estorbar.
            with st.expander("Show executed code"):
                st.code(response.last_code_executed, language="python")
            st.session_state.messages.append({"role": "assistant", "content": response_content})
