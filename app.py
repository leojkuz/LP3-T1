import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM
from data import load_data
import io
from PIL import Image

# CAMBIO 1: Ajustamos el layout a 'centered' (centrado) en vez de 'wide'.
st.set_page_config(layout="centered")
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

# ÚNICA FUENTE DE VERDAD: Este bucle ahora manejará las pestañas.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else: # Si el mensaje es del asistente...
            # CAMBIO 2: Lógica de pestañas integrada aquí.
            tab_res, tab_code = st.tabs(["Resultado", "Código"])
            with tab_res:
                if message.get("type") == "dataframe":
                    st.dataframe(message["content"], use_container_width=True, hide_index=True)
                elif message.get("type") == "chart":
                    st.image(message["content"])
                else:
                    st.write(message["content"])
            with tab_code:
                # Usamos .get("code", "...") para ser seguros si un mensaje antiguo no tuviera código.
                st.code(message.get("code", "No code executed for this response."), language="python")

# Lógica del Chat
if prompt := st.chat_input("🗣️ Chat with Dataframe"):
    st.session_state.messages.append({"role": "user", "content": prompt, "type": "string"})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            history = st.session_state.messages[-10:]
            context_string = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in history if msg['type'] == 'string']
            )
            full_prompt = f"Based on the following conversation history:\n{context_string}\n\nAnswer this new question: {prompt}"

            response = st.session_state.sdf_instance.chat(full_prompt)

            # CAMBIO 3: Ahora también guardamos el código ejecutado en el historial.
            if response.type == "dataframe":
                response_message = {"role": "assistant", "content": response.value, "type": "dataframe", "code": response.last_code_executed}
            elif response.type == "chart":
                response_message = {"role": "assistant", "content": response.value, "type": "chart", "code": response.last_code_executed}
            else:
                response_message = {"role": "assistant", "content": response.value, "type": "string", "code": response.last_code_executed}

            st.session_state.messages.append(response_message)
            st.rerun()
