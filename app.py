import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

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

herramientas = [
    Tool(
        name="buscar_en_documento",
        func=tool_rag,
        description="Útil para responder preguntas sobre el contenido del documento PDF. Úsala siempre como primera opción."
    ),
    Tool(
        name="buscar_en_internet",
        func=DuckDuckGoSearchRun(),
        description="Útil para datos actuales o si la información no está en el documento."
    )
]

# guardarrail
def es_relevante(pregunta):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Le pedimos que sea binario para no fallar
    prompt = f"Determina si la siguiente pregunta tiene relación con el tema de Inteligencia Artificial: '{pregunta}'. Responde solo 'relevante' o 'irrelevante'."
    respuesta = llm.invoke(prompt).content.lower()
    return "relevante" in respuesta

# creacion del agente
llm_agente = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# Obtenemos un prompt estándar para agentes ReAct
prompt_react = hub.pull("hwchase17/react")
agente = create_react_agent(llm_agente, herramientas, prompt_react)
agente_executor = AgentExecutor(agent=agente, tools=herramientas, verbose=True)

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
                    resultado = agente_executor.invoke({"input": prompt})
                    respuesta_final = resultado["output"]
            
            st.markdown(respuesta_final)
    
    st.session_state.messages.append({"role": "assistant", "content": respuesta_final})