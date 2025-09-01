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

# --- Configuración Global del LLM y PandasAI ---
llm = LiteLLM(
    model="gemini/gemini-2.5-flash",
    api_key=st.secrets["gemini_key"],
    temperature=0.5
)

pai.config.set({
    "llm": llm,
    'history_size': 10
})

# --- Carga de datos ---
df = load_data("./data")
sdf = SmartDataframe(df)

with st.expander("🔎 Dataframe Preview"):
    st.write(df.tail(3))

# --- GESTIÓN DEL HISTORIAL VISUAL ---
# NUEVO: Inicializamos el historial en st.session_state si no existe.
if "messages" not in st.session_state:
    st.session_state.messages = []

# NUEVO: Mostramos todos los mensajes antiguos al principio de cada ejecución.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Lógica del Chat ---
if prompt := st.chat_input("🗣️ Chat with Dataframe"):

    # NUEVO: Añadimos el mensaje del usuario al historial para que se guarde.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Lo mostramos en la pantalla.
    with st.chat_message("user"):
        st.write(prompt)

    # Mostramos la respuesta de la IA.
    with st.chat_message("assistant"):
        response = sdf.chat(prompt)
        response_content = response.value # Guardamos el contenido para el historial

        # La misma lógica de antes para mostrar la respuesta actual.
        if response.type == "dataframe":
            st.dataframe(response_content, use_container_width=True, hide_index=True)
        elif response.type == "chart":
            st.image(response_content)
        else:
            st.write(response_content)

        # Mostramos el código ejecutado en un desplegable para no estorbar.
        with st.expander("Show executed code"):
            st.code(response.last_code_executed, language="python")

        # NUEVO: Añadimos la respuesta de la IA al historial para que se guarde.
        # Nota: Guardamos la respuesta simple para mantener el historial visual limpio.
        # El código es más complejo para guardar también tablas e imágenes, podemos verlo después.
        st.session_state.messages.append({"role": "assistant", "content": response_content})

