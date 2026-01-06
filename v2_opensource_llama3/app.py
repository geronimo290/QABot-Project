"""
### qabot_v2 ###

"""

#Instalacion de librerias 
## en termial pip install -r requirements.txt

# --- FIX PARA CHROMA DB EN HUGGING FACE ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# -------------------------------------------

#Importacion de librerias
import os
# # Impoprtacion de librerias Groq y HuggingFace
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
# Importacion de liberias LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr

#Suprimir Warnings

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# API KEY Groq
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

### Cambios en comparacion al qabot Watsonx ###

### LLM Llama 3 via Groq ###

def get_llm():

    #Configuracion del modelo
    llm_groq = ChatGroq(
        model_name="llama-3.3-70b-versatile",

        #Temperatura: número (generalmente entre 0 y 2) que ajusta cuánta aleatoriedad usa el modelo al elegir la siguiente palabra

        # + temp mas baja (0 - 0.3) mas presiso, logico y reptible
        # + temp media (0.5 - 0.8) es un equilibrio entre coherencia y creatividad
        # + temp alta (1.0 - 2.0) mayor creatividad, impredisible e incluso caotico
        temperature=0.1, 

        # Max Tokens: Cantidad de texto que puede generar como maximo
        max_tokens=500,

        model_kwargs={
            "top_p" : 0.9, #Ayuda a la precision
        }
    )
    return llm_groq


# Document Loader
def document_loader(file):
    #Inicializar el cargador con nombre o ruta del archivo
    #file.name otiene la ruta temporal donde gradio guardo el PDF que subio
    loader = PyPDFLoader(file.name)

    #Carga de contenido, devulve lista de objetos Document (uno por pagina)
    loaded_document = loader.load()
    return loaded_document

    #El cargador carga el contenido, pero su metodo .load(), NO LO DIVIDE EN FRACMENTOS
    #Se define un divisor de documentos

#Text Splitter
def text_splitter(data):
    """
    Recibe una lista de documentos cargados (data) y devuelve una lista de fragmentos (chunks).
    """
    #Es el más inteligente: intenta cortar por párrafos, luego frases, luego palabras, para no romper ideas.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    # Ejecucion del corte
    # .split_documents toma la lista de documentos (con metadatos incluidos)
    chunks = text_splitter.split_documents(data)

    return chunks

### EMBEDDINGS HuggingFace Local ###
def embedding_generator():
    """
    Usa un modelo pequeño que corre en tu CPU (gratis y sin internet).
    """
    # "all-MiniLM-L6-v2" es el estándar de oro para RAG ligero y rápido.
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model

def vector_database(chunks):
    """
    Toma los fragmentos de texto y el modelo de embeddings, y crea la base de datos vectorial.
    """

    embedding_model = embedding_generator()
    vectordb = Chroma.from_documents(
        documents=chunks, # Los trozos de texto
        embedding=embedding_model # El modelo de embedding convertidor
    )
    return vectordb

# La funcion retriever conecta la CARGA, DIVISION Y BASE DE DATOS, en una sola PIPLINE automatica.

##Retriever
def retriever(file):
    # Carga de archivo (T1)
    split = document_loader(file)

    # Division del texto (T2)
    # pasamos split como entrada
    chunks = text_splitter(split)

    # Creacion de la base de datos e embeddings de los vectores (T4)
    # pasamos chunks como entrada
    vectordb = vector_database(chunks)

    # Se convierte la base de datos en un buscador de archivos
    retriever = vectordb.as_retriever()

    return retriever


## QA Chain
def retriever_qa(file, query):
    print("Starting retriever_qa function...")
    #Llamamos al cerebro LLM
    llm = get_llm()
    print("LLM initialized.")

    #Llamamos al Retriever (Buscador)
    retriever_obj = retriever(file) #(T5)
    print("Retriever object created.")

    #Creacion de la cadena RAG (RetrieverQA)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", #"stuff" mete el texto en el prompt
        retriever=retriever_obj, #Usamos el objeto retriever creado arriba
        return_source_documents=False #False para que solo devuelva texto (limpio)
    )
    print("RetrievalQA chain created.")

    #Ejecucion de la pregunta, se pasa query (pregunta) al modelo invoke
    response = qa.invoke(query)
    print(f"Response from QA chain: {response}")

    #Resultado del texto
    return response["result"]

if __name__ == "__main__":
# --- Interfaz Gradio  ---
    rag_application = gr.Interface(
    fn=retriever_qa,
    inputs=[
        gr.File(label="Sube tu PDF", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Tu Pregunta", lines=2, placeholder="Ej: ¿Cuál es la conclusión principal del documento?")
    ],
    outputs=gr.Textbox(label="Respuesta de Llama 3"),
    title="RAG Chatbot con Llama 3",
    description="Analiza documentos PDF usando Llama 3 (Groq) y Embeddings locales. 100% Open Source.",
)


    rag_application.launch()