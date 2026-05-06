import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent

load_dotenv()


st.set_page_config(page_title="Agente Experto", layout="centered")
st.title("Agente Especializado")

# Cargamos la base de datos que creamos con ingestion.py
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


def tool_rag(query):
    # Busca en el PDF y devuelve los textos encontrados
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

buscador_web = DuckDuckGoSearchRun()

herramientas = [
    Tool(
        name="buscar_en_documento",
        func=tool_rag,
        description="""Útil para buscar información en el documento PDF local. 
        IMPORTANTE: El documento está escrito en INGLÉS. Si el usuario te pregunta en español, 
        DEBES traducir la consulta de búsqueda al inglés antes de usar esta herramienta."""
    ),
    Tool(
        name="buscar_en_internet",
        func=buscador_web.invoke,
        description="Útil como segunda opción si la información no está en el documento local."
    )
]

# guardarrail

def es_relevante(pregunta):
    llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0)
    
    tema = "sensores inerciales, biomecánica y seguimiento del movimiento articular"
    
    # Volvemos a las palabras exactas del enunciado
    prompt = f"""
    Determina si la siguiente pregunta tiene relación con el tema de '{tema}'. 
    Responde ÚNICAMENTE con la palabra 'relevante' si tiene relación, o con la palabra 'irrelevante' si no la tiene.
    Pregunta: '{pregunta}'
    """
    
    # Limpiamos espacios y forzamos minúsculas para que la comparación no falle
    respuesta = llm.invoke(prompt).content.strip().lower()
    
    print("\n--- DEBUG DEL GUARDARRAÍL ---")
    print(f"Pregunta del usuario: {pregunta}")
    print(f"Decisión del LLM: '{respuesta}'")
    print("-----------------------------\n")
    
    return respuesta == "relevante"

# creacion del agente
llm_agente = ChatOpenAI(model="gpt-5.4-nano", temperature=0)
# Obtenemos un prompt estándar para agentes ReAct
instrucciones_experto = """
Eres un asistente experto en biomecánica, sensores inerciales y seguimiento del movimiento articular.
Tu objetivo es responder de forma rigurosa y técnica.

REGLAS ESTRICTAS:
1. Usa SIEMPRE primero la herramienta 'buscar_en_documento'.
2. Si la información no está ahí, usa 'buscar_en_internet'.
3. OBLIGATORIO: Al final de tu respuesta, debes indicar claramente la fuente usada escribiendo "[Fuente: Documento Local]" o "[Fuente: Internet]".
"""

agente = create_react_agent(llm_agente, tools=herramientas, prompt=instrucciones_experto)

# STREAMLIT - interfaz
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Entrada del usuario
if prompt := st.chat_input("¿Qué quieres saber?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Guardarraíl
        if not es_relevante(prompt):
            respuesta_final = "Lo siento, soy un agente especializado y esa pregunta está fuera de mi dominio."
            st.warning(respuesta_final)
        else:
            # El Agente decide qué hacer
            with st.spinner("Pensando..."):
                # Usamos un expander para mostrar el razonamiento (Chain of Thought)
                with st.expander("Ver razonamiento del agente"):             
                    inputs = {"messages": st.session_state.messages}
                    resultado = agente.invoke(inputs)
                    respuesta_final = resultado["messages"][-1].content
            
            st.markdown(respuesta_final)
    
    st.session_state.messages.append({"role": "assistant", "content": respuesta_final})