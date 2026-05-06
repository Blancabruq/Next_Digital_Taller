import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def preparar_db():
    # 1. Cargar el PDF (cambia el nombre al de tu archivo en la carpeta /data)
    loader = PyPDFLoader("./data/Upper_limb_joint_angle_tracking_with_inertial_sensors.pdf")
    documents = loader.load()

    # 2. Partir el texto en trozos (chunks)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # 3. Crear Embeddings y guardar en Chroma
    # Los embeddings convierten texto en números (vectores)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory="./chroma_db" # Aquí se guardará físicamente
    )
    print( "Base de datos creada y guardada en ./chroma_db")

if __name__ == "__main__":
    preparar_db()