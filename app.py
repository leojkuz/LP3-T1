import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM
from data import load_data
import io
from PIL import Image

st.set_page_config(layout="wide")
st.write("# Chat with Credit Card Fraud Dataset 🦙")

llm = LiteLLM(
    model="gemini/gemini-2.5-flash",
    api_key=st.secrets["gemini_key"],
    temperature=0.5
)

df = load_data("./data")

if "sdf_instance" not in st.session_state:
    st.session_state.sdf_instance = SmartDataframe(df, config={"llm": llm})

with st.expander("🔎 Dataframe Preview"):
    st.write(df.tail(3))

if "messages" not in st.session_state:
    st.session_state.messages = []

# ÚNICA FUENTE DE VERDAD: Este bucle es el responsable de dibujar todo el chat.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("type") == "dataframe":
            st.dataframe(message["content"], use_container_width=True, hide_index=True)
        elif message.get("type") == "chart":
            st.image(message["content"])
        else:
            st.write(message["content"])

# Lógica del Chat
if prompt := st.chat_input("🗣️ Chat with Dataframe"):
    # Añadimos y mostramos el mensaje del usuario inmediatamente.
    st.session_state.messages.append({"role": "user", "content": prompt, "type": "string"})
    with st.chat_message("user"):
        st.write(prompt)

    # Creamos el contenedor de la IA, pero no dibujamos la respuesta final aquí.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Lógica de contexto manual
            history = st.session_state.messages[-10:]
            context_string = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in history if msg['type'] == 'string']
            )
            full_prompt = f"Based on the following conversation history:\n{context_string}\n\nAnswer this new question: {prompt}"

            response = st.session_state.sdf_instance.chat(full_prompt)

            # Preparamos el mensaje de respuesta
            if response.type == "dataframe":
                response_message = {"role": "assistant", "content": response.value, "type": "dataframe"}
            elif response.type == "chart":
                response_message = {"role": "assistant", "content": response.value, "type": "chart"}
            else:
                response_message = {"role": "assistant", "content": response.value, "type": "string"}

            # Añadimos la respuesta al historial
            st.session_state.messages.append(response_message)

            # LA SOLUCIÓN: Forzamos una re-ejecución del script.
            st.rerun()
