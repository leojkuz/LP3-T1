import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM
from data import load_data
import io
from PIL import Image

# --- Configuración Streamlit y Título ---
st.set_page_config(layout="wide")
st.write("# Chat with Credit Card Fraud Dataset 🦙")

# --- Configuración del LLM ---
llm = LiteLLM(
    model="gemini/gemini-2.5-flash",
    api_key=st.secrets["gemini_key"],
    temperature=0.5
)

# --- Carga de datos ---
df = load_data("./data")

# --- Inicialización del Agente ---
if "sdf_instance" not in st.session_state:
    st.session_state.sdf_instance = SmartDataframe(
        df,
        config={ "llm": llm }
    )

with st.expander("🔎 Dataframe Preview"):
    st.write(df.tail(3))

# --- Gestión del Historial Visual (AHORA MÁS INTELIGENTE) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostramos el historial. Esta parte ahora entiende los diferentes tipos de contenido.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Si el mensaje guardado fue un dataframe, lo mostramos como tal.
        if message.get("type") == "dataframe":
            st.dataframe(message["content"], use_container_width=True, hide_index=True)
        # Si fue un gráfico, lo mostramos como imagen.
        elif message.get("type") == "chart":
            st.image(message["content"])
        # Si no, es texto normal.
        else:
            st.write(message["content"])

# --- Lógica del Chat ---
if prompt := st.chat_input("🗣️ Chat with Dataframe"):
    # Añadimos y mostramos el mensaje del usuario (esto es solo texto, así que es simple)
    st.session_state.messages.append({"role": "user", "content": prompt, "type": "string"})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # --- Lógica de Construcción de Contexto Manual ---
            history = st.session_state.messages[-10:]
            context_messages = [f"{msg['role']}: {msg['content']}" for msg in history]
            # Filtramos los mensajes que no son texto plano para no confundir a la IA.
            # Solo pasamos la conversación real como contexto.
            context_string = "\n".join(
                [f"user: {msg['content']}" for msg in history if msg['role'] == 'user']
            )
            full_prompt = f"Based on the following conversation history:\n{context_string}\n\nAnswer this new question: {prompt}"

            # --- Llamada a la IA y Manejo de Respuestas (LA PARTE RESTAURADA) ---
            response = st.session_state.sdf_instance.chat(full_prompt)

            # Restauramos la lógica para manejar cada tipo de respuesta
            if response.type == "dataframe":
                st.dataframe(response.value, use_container_width=True, hide_index=True)
                st.session_state.messages.append({"role": "assistant", "content": response.value, "type": "dataframe"})

            elif response.type == "chart":
                st.image(response.value)
                st.session_state.messages.append({"role": "assistant", "content": response.value, "type": "chart"})

            else:
                st.write(response.value)
                st.session_state.messages.append({"role": "assistant", "content": response.value, "type": "string"})

            # El código ejecutado siempre va al final, en un desplegable
            with st.expander("Show executed code"):
                st.code(response.last_code_executed, language="python")
