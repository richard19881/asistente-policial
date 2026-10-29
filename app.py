import time
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# Nombre oficial y vigente del modelo en la API
MODELO_IA = "gemini-3.8-flash"

INSTRUCCIONES = """Eres un asistente de orientación sobre procedimientos policiales en Ecuador.
Reglas obligatorias:
1. Responde SOLO con base en los artículos entregados en el CONTEXTO. No uses otro conocimiento legal externo.
2. Si el contexto no alcanza para responder con seguridad, di claramente: "No encuentro sustento normativo suficiente en la normativa cargada" y sugiere consultar a fiscalía o asesoría jurídica.
3. Cita siempre la norma y el artículo (por ejemplo: COIP, Art. X).
4. Si faltan datos clave (edad, tipo de orden, quién la emitió, etc.), indica qué dato falta y cómo cambiaría la respuesta.
5. Usa strictly esta estructura en tu respuesta:
   - **Hechos relevantes**
   - **Base legal**
   - **Procedimiento sugerido**
   - **Alertas y excepciones**
   - **Cuándo consultar a fiscalía o asesoría jurídica**
6. Eres una herramienta de orientación: no reemplazas el criterio del servidor policial ni la asesoría jurídica oficial."""

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente Normativo Policial", page_icon="🛡️")

# --- ESTILOS VISUALES INSTITUCIONALES (CSS) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0b131d;
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] {
        background-color: #0d1b2a;
        border-right: 1px solid #c59b27;
    }
    h1, h2, h3 {
        color: #f0f0f0 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stChatMessage {
        border-radius: 8px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Título y cabecera
st.title("🛡️ Asistente Normativo Policial (Ecuador)")
st.caption("Herramienta de orientación basada en 27 cuerpos normativos institucionales. No sustituye asesoría jurídica.")

# --- PANEL LATERAL ---
try:
    st.sidebar.image("escudo.png", use_container_width=True)
except Exception:
    st.sidebar.markdown("### 🛡️ POLICÍA NACIONAL")

st.sidebar.markdown("---")
clave = st.sidebar.text_input("Clave de Google AI Studio", type="password")
st.sidebar.info("Pega aquí tu clave API generada en Google AI Studio para activar la inteligencia artificial.")

@st.cache_resource
def cargar_recursos():
    modelo = SentenceTransformer("intfloat/multilingual-e5-small")
    cliente = chromadb.PersistentClient(path="base_vectorial")
    coleccion = cliente.get_collection("normativa")
    return modelo, coleccion

modelo, coleccion = cargar_recursos()

pregunta = st.chat_input("Describe el caso o escribe tu pregunta (ej: uso de la fuerza, aprehensión, tránsito, etc.)...")

if pregunta:
    if not clave:
        st.warning("⚠️ Por favor, pega tu clave de Google AI Studio en el panel de la izquierda para continuar.")
        st.stop()
        
    with st.chat_message("user"):
        st.write(pregunta)
        
    # Buscar en la base vectorial los 8 fragmentos más parecidos
    vector = modelo.encode(["query: " + pregunta], normalize_embeddings=True).tolist()
    resultado = coleccion.query(query_embeddings=vector, n_results=8)
    
    fragmentos = resultado.get("documents", [[]])[0]
    metadatas = resultado.get("metadatas", [[]])[0]
    
    # Construcción segura del contexto
    contexto_list = []
    for d, m in zip(fragmentos, metadatas):
        meta = m if isinstance(m, dict) else {}
        norma = meta.get("norma", "Norma Institucional")
        articulo = meta.get("articulo", "Artículo General")
        texto_doc = str(d) if d is not None else ""
        contexto_list.append(f"[{norma} - {articulo}]\n{texto_doc}")
        
    contexto = "\n\n".join(contexto_list)
    mensaje = f"CONTEXTO:\n{contexto}\n\nPREGUNTA DEL USUARIO:\n{pregunta}"
    
    with st.chat_message("assistant"):
        with st.spinner("Analizando los 27 cuerpos normativos..."):
            cliente = genai.Client(api_key=clave)
            exito = False
            
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
                    if intento < 2:
                        time.sleep(2)
                    else:
                        st.error(f"⚠️ Error de conexión con la IA o saturación momentánea. Detalle: {str(e)}")
                        break
            
            if exito:
                with st.expander("Ver artículos consultados en la base de datos"):
                    for d, m in zip(fragmentos, metadatas):
                        meta = m if isinstance(m, dict) else {}
                        norma = meta.get("norma", "Norma Institucional")
                        articulo = meta.get("articulo", "Artículo General")
                        texto_doc = str(d) if d is not None else ""
                        st.markdown(f"**Norma: {norma} | Artículo: {articulo}**")
                        st.text(texto_doc[:700])