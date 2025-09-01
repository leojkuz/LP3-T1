import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM
from data import load_data
import io
from PIL import Image
import base64 # NUEVA LIBRERÍA: para "traducir" imágenes a texto.

# --- Función Auxiliar para codificar la imagen ---
# La he creado para mantener el código principal más limpio.
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

st.set_page_config(layout="centered")
st.write("# Chat with Credit Card Fraud Dataset 🦙")

llm = LiteLLM(
    model="gemini/gemini-2.5-flash", # Modelo con capacidad de visión
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

# Este bucle sigue igual, se encarga de la parte VISUAL.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            tab_res, tab_code = st.tabs(["Resultado", "Código"])
            with tab_res:
                if message.get("type") == "dataframe":
                    st.dataframe(message["content"], use_container_width=True, hide_index=True)
                elif message.get("type") == "chart":
                    # Muestra la imagen desde la ruta del archivo guardada
                    st.image(message["content"])
                else:
                    st.write(message["content"])
            with tab_code:
                st.code(message.get("code", "No code executed."), language="python")

if prompt := st.chat_input("🗣️ Chat with Dataframe"):
    st.session_state.messages.append({"role": "user", "content": prompt, "type": "string"})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # --- NUEVA LÓGICA DE CONSTRUCCIÓN DE CONTEXTO MULTIMODAL ---
            # Ya no es un simple string, es una lista de mensajes que la IA procesará secuencialmente.
            conversation_history = []
            for msg in st.session_state.messages[-10:]: # Tomamos los últimos mensajes
                if msg["type"] == "string":
                    conversation_history.append(f"{msg['role']}: {msg['content']}")
                # ¡LA MAGIA! Si un mensaje fue un gráfico, añadimos su "memoria visual" al contexto.
                elif msg["type"] == "chart":
                    conversation_history.append(f"assistant: [Generated a chart. User can now see it.]")

            # Combinamos el historial con la nueva pregunta
            full_prompt = "\n".join(conversation_history)

            # Para la visión, pasamos la imagen como un argumento separado si existe.
            image_context = None
            # Buscamos el último gráfico generado en la conversación
            for msg in reversed(st.session_state.messages):
                if msg.get("type") == "chart":
                    image_path = msg["content"]
                    if os.path.exists(image_path):
                        image_context = image_path
                        break

            # Llamamos a chat, pero ahora con un posible contexto de imagen
            response = st.session_state.sdf_instance.chat(
                full_prompt,
                image=image_context # La mayoría de implementaciones de LLM aceptan un kwarg así
            )

            # El resto de la lógica se mantiene, pero ahora guardaremos también la data de la imagen
            if response.type == "dataframe":
                response_message = {"role": "assistant", "content": response.value, "type": "dataframe", "code": response.last_code_executed}
            elif response.type == "chart":
                # Codificamos la imagen para la memoria de la IA
                base64_image = encode_image(response.value)
                response_message = {"role": "assistant", "content": response.value, "type": "chart", "code": response.last_code_executed", "base64": base64_image}
            else:
                response_message = {"role": "assistant", "content": response.value, "type": "string", "code": response.last_code_executed}

            st.session_state.messages.append(response_message)
            st.rerun()
