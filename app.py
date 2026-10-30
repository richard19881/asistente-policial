import time
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# Nombre oficial y vigente del modelo en la API
MODELO_IA = "gemini-3.8-flash"

INSTRUCCIONES = """Eres un asistente de orientación sobre procedimientos policiales en Ecuador, especializado en asesoría táctico-operativa y legal.

Reglas obligatorias:
1. Responde SOLO con base en los artículos entregados en el CONTEXTO. No inventes artículos ni uses leyes externas a la base.
2. Si el contexto no alcanza para responder con seguridad, indica claramente: "No encuentro sustento normativo suficiente en la normativa cargada" y sugiere consultar a fiscalía o asesoría jurídica.
3. Cita siempre la norma y el artículo correspondiente (ejemplo: COIP, Art. 530; Ley Uso Legítimo de la Fuerza, Art. X).
4. RECOMENDACIONES OPERATIVAS Y BUENAS PRÁCTICAS POLICIALES (OBLIGATORIO):
   - En temas de BOLETAS (aprehensión, auxilio, citación, encarcelamiento):
     * Recomienda SIEMPRE verificar la VIGENCIA en sistemas oficiales (eSATJE, SIIPNE, Módulo de Boletas) antes de ejecutar.
     * En el Parte Policial, recordar adjuntar copia legible de la boleta, firma/fe de recepción de la autoridad o custodio, y copia de cédula del aprehendido/notificado.
     * Si un tercero entrega la boleta a la policía: Aclarar si la normativa exige que la boleta sea presentada por la víctima/titular o si la fuerza pública puede actuar por requerimiento directo del órgano judicial competente.
   - Detalla qué hacer si la persona notificada/aprehendida se niega a firmar o si existen inconsistencias en nombres/números de cédula.
5. Usa estrictamente esta estructura en tu respuesta:
   - **Hechos relevantes y tipificación inicial**
   - **Base legal aplicable**
   - **Procedimiento sugerido paso a paso**
   - **Recomendaciones operativas y formalidades del Parte Policial** (verificaciones de vigencia, firmas, anexos obligatorios)
   - **Alertas, excepciones y legalidad ante terceros**
   - **Cuándo consultar a Fiscalía o Asesoría Jurídica**
6. Eres una herramienta de orientación: no reemplazas el criterio del servidor policial ni la asesoría jurídica oficial."""

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente Normativo Policial", page_icon="🛡️", layout="wide")

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
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Título y cabecera
st.title("🛡️ Asistente Normativo Policial (Ecuador)")
st.caption("Herramienta de orientación basada en 27 cuerpos normativos institucionales. Gestión de casos en tiempo real.")

# --- PANEL LATERAL ---
try:
    st.sidebar.image("escudo.png", use_container_width=True)
except Exception:
    st.sidebar.markdown("### 🛡️ POLICÍA NACIONAL DEL ECUADOR")

st.sidebar.markdown("---")
clave = st.sidebar.text_input("Clave de Google AI Studio", type="password")
st.sidebar.info("Pega aquí tu clave API generada en Google AI Studio para activar el asistente.")

# Botón para cerrar caso e iniciar nueva consulta
if st.sidebar.button("🔴 Cerrar caso e iniciar nueva consulta", use_container_width=True):
    st.session_state["messages"] = []
    st.rerun()

# --- CARGA MOPTINIZADA DE RECURSOS ---
@st.cache_resource
def cargar_recursos():
    # Modelo ligero optimizado para embeddings rápidos
    modelo = SentenceTransformer("intfloat/multilingual-e5-small")
    cliente = chromadb.PersistentClient(path="base_vectorial")
    coleccion = cliente.get_collection("normativa")
    return modelo, coleccion

modelo, coleccion = cargar_recursos()

# --- GESTIÓN DEL HISTORIAL DE CONVERSACIÓN (MEMORIA POR CASO) ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Mostrar el historial de la conversación actual del caso
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

pregunta = st.chat_input("Escribe tu consulta o continúa preguntando sobre el mismo caso...")

if pregunta:
    if not clave:
        st.warning("⚠️ Por favor, ingresa tu clave de Google AI Studio en el panel izquierdo.")
        st.stop()

    # Guardar y mostrar la pregunta del usuario
    st.session_state["messages"].append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.write(pregunta)

    # 1. Búsqueda vectorial rápida de contexto (retornamos 6 fragmentos para acelerar la respuesta)
    vector = modelo.encode(["query: " + pregunta], normalize_embeddings=True).tolist()
    resultado = coleccion.query(query_embeddings=vector, n_results=6)
    
    fragmentos = resultado.get("documents", [[]])[0]
    metadatas = resultado.get("metadatas", [[]])[0]
    
    # Construcción limpia del contexto normativo
    contexto_list = []
    for d, m in zip(fragmentos, metadatas):
        meta = m if isinstance(m, dict) else {}
        norma = meta.get("norma", "Norma Institucional")
        articulo = meta.get("articulo", "Artículo General")
        texto_doc = str(d) if d is not None else ""
        contexto_list.append(f"[{norma} - {articulo}]\n{texto_doc}")
        
    contexto = "\n\n".join(contexto_list)

    # 2. Reconstrucción del historial para enviar a Gemini (permite repreguntar sobre el mismo caso)
    historial_texto = ""
    for m in st.session_state["messages"]:
        rol = "POLICÍA (Usuario)" if m["role"] == "user" else "ASISTENTE"
        historial_texto += f"\n{rol}: {m['content']}\n"

    prompt_completo = f"CONTEXTO LEGAL EXTRAÍDO DE LA BASE VECTORIAL:\n{contexto}\n\nHISTORIAL Y CASO ACTUAL:\n{historial_texto}"

    # 3. Generación de respuesta con Gemini
    with st.chat_message("assistant"):
        with st.spinner("Consultando la normativa y formulando recomendaciones de procedimiento..."):
            cliente = genai.Client(api_key=clave)
            exito = False
            
            for intento in range(3):
                try:
                    respuesta = cliente.models.generate_content(
                        model=MODELO_IA,
                        contents=prompt_completo,
                        config=types.GenerateContentConfig(system_instruction=INSTRUCCIONES),
                    )
                    texto_respuesta = respuesta.text
                    st.write(texto_respuesta)
                    
                    # Guardar respuesta en el historial del caso
                    st.session_state["messages"].append({"role": "assistant", "content": texto_respuesta})
                    exito = True
                    break
                except Exception as e:
                    if intento < 2:
                        time.sleep(1)
                    else:
                        st.error(f"⚠️ Error al conectar con la IA: {str(e)}")
                        break
            
            if exito:
                with st.expander("Ver artículos y fragmentos legales consultados"):
                    for d, m in zip(fragmentos, metadatas):
                        meta = m if isinstance(m, dict) else {}
                        norma = meta.get("norma", "Norma Institucional")
                        articulo = meta.get("articulo", "Artículo General")
                        texto_doc = str(d) if d is not None else ""
                        st.markdown(f"**Norma: {norma} | Artículo: {articulo}**")
                        st.text(texto_doc[:600])