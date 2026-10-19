import time
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# Nombre del modelo estable de Google AI Studio
MODELO_IA = "gemini-1.5-flash"

INSTRUCCIONES = """Eres un asistente de orientación sobre procedimientos policiales en Ecuador.
Reglas obligatorias:
1. Responde SOLO con base en los artículos entregados en el CONTEXTO. No uses otro conocimiento legal externo.
2. Si el contexto no alcanza para responder con seguridad, di claramente: "No encuentro sustento normativo suficiente en la normativa cargada" y sugiere consultar a fiscalía o asesoría jurídica.
3. Cita siempre la norma y el artículo (por ejemplo: COIP, Art. X).
4. Si faltan datos clave (edad, tipo de orden, quién la emitió, etc.), indica qué dato falta y cómo cambiaría la respuesta.
5. Usa estrictamente esta estructura en tu respuesta:
   - **Hechos relevantes**
   - **Base legal**
   - **Procedimiento sugerido**
   - **Alertas y excepciones**
   - **Cuándo consultar a fiscalía o asesoría jurídica**
6. Eres una herramienta de orientación: no reemplazas el criterio del servidor policial ni la asesoría jurídica oficial."""

st.set_page_config(page_title="Asistente Normativo Policial", page_icon="🛡️")
st.title("🛡️ Asistente Normativo Policial (Ecuador)")
st.caption("Herramienta de orientación basada en normativa pública. No sustituye asesoría jurídica.")

# Panel lateral para la clave
clave = st.sidebar.text_input("Clave de Google AI Studio", type="password")
st.sidebar.info("Pega aquí tu clave API generada en Google AI Studio para activar la inteligencia artificial.")

@st.cache_resource
def cargar_recursos():
    modelo = SentenceTransformer("intfloat/multilingual-e5-small")
    cliente = chromadb.PersistentClient(path="base_vectorial")
    coleccion = cliente.get_collection("normativa")
    return modelo, coleccion

modelo, coleccion = cargar_recursos()

pregunta = st.chat_input("Describe el caso o escribe tu pregunta (ej: allanamientos, boletas, etc.)...")

if pregunta:
    if not clave:
        st.warning("⚠️ Por favor, pega tu clave de Google AI Studio en el panel de la izquierda para continuar.")
        st.stop()
        
    with st.chat_message("user"):
        st.write(pregunta)
        
    # Buscar en la base vectorial los 6 fragmentos más parecidos
    vector = modelo.encode(["query: " + pregunta], normalize_embeddings=True).tolist()
    resultado = coleccion.query(query_embeddings=vector, n_results=6)
    fragmentos = resultado["documents"][0]
    metadatos = resultado["metadatas"][0]
    
    contexto = "\n\n".join(
        f"[{m['norma']} - {m['articulo']}]\n{d}" for d, m in zip(fragmentos, metadatos)
    )
    
    mensaje = f"CONTEXTO:\n{contexto}\n\nPREGUNTA DEL USUARIO:\n{pregunta}"
    
    with st.chat_message("assistant"):
        with st.spinner("Analizando la normativa ecuatoriana..."):
            cliente = genai.Client(api_key=clave)
            exito = False
            
            # Reintentos automáticos para manejar picos de saturación (Error 503)
            for intento in range(3):
                try:
                    respuesta = cliente.models.generate_content(
                        model=MODELO_IA,
                        contents=mensaje,
                        config=types.GenerateContentConfig(system_instruction=INSTRUCCIONES),
                    )
                    st.write(respuesta.text)
                    exito = True
                    break
                except Exception as e:
                    if "503" in str(e) and intento < 2:
                        time.sleep(3)  # Espera 3 segundos antes de reintentar
                    else:
                        st.error(f"⚠️ Servidores de Google con alta demanda temporal o error de conexión. Por favor, reintenta en un momento.")
                        break
            
            if exito:
                with st.expander("Ver artículos consultados en la base de datos"):
                    for d, m in zip(fragmentos, metadatos):
                        st.markdown(f"**Norma: {m['norma']} | Artículo: {m['articulo']}**")
                        st.text(d[:700])