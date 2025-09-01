
import os
import streamlit as st
import pandas as pd
import pandasai as pai
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM
from data import load_data

# --- Configuración Streamlit ---
st.write("# Chat with Credit Card Fraud Dataset 🦙")

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
# MODIFICADO: Creamos el SmartDataframe una sola vez, al inicio.
sdf = SmartDataframe(df)

with st.expander("🔎 Dataframe Preview"):
    st.write(df.tail(3))

# --- Lógica del Chat ---
# MODIFICADO: Reemplazamos el text_area por chat_input, como en el ejemplo.
if prompt := st.chat_input("🗣️ Chat with Dataframe"):

    # Mostramos el mensaje del usuario en una burbuja de chat.
    with st.chat_message("human"):
        st.write(prompt)

    # Mostramos la respuesta de la IA en su propia burbuja.
    with st.chat_message("assistant"):
        # Ejecutamos la consulta.
        response = sdf.chat(prompt)

        # Manejamos los diferentes tipos de respuesta, como en tu código original pero
        # con la presentación de pestañas del segundo ejemplo.
        if response.type == "dataframe":
            tab_res, tab_code = st.tabs(["Resultado", "Código"])
            with tab_res:
                st.dataframe(response.value, use_container_width=True, hide_index=True)
            with tab_code:
                st.code(response.last_code_executed, language="python")

        elif response.type == "chart":
            # Lógica para manejar la imagen del gráfico, copiada del ejemplo funcional.
            with open(response.value, "rb") as f:
                img_bytes = f.read()
            img = Image.open(io.BytesIO(img_bytes))

            tab_res, tab_code = st.tabs(["Resultado", "Código"])
            with tab_res:
                st.image(img)
            with tab_code:
                st.code(response.last_code_executed, language="python")

            os.remove(response.value) # Buena práctica del ejemplo: borrar el archivo temporal.

        else:
            tab_res, tab_code = st.tabs(["Resultado", "Código"])
            with tab_res:
                st.write(response.value)
            with tab_code:
                st.code(response.last_code_executed, language="python")
