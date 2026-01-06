# Impoprtacion de librerias Watsonx
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
# Importacion de liberias LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
#Imoptacionde HuggingFace y Gradio
from huggingface_hub import HfFolder
import gradio as gr
#Suprimir Warnings

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

## LLM IBMWatsonx ##



def get_llm():

  # CONFIGURACION DE PARAMETROS DEL MODELO (como se controla el modelo)
  parametros = {
    #Decodificacion "greedy" elige la palabra mas probable (codiciosa)
    #Mejor para respuestas serias. ('sample' es mejor para creatividad.)
    GenParams.DECODING_METHOD: "greedy",

    #Cantidad de texto que puede generar como maximo
    GenParams.MAX_NEW_TOKENS: 200,

    #Temperatura: número (generalmente entre 0 y 2) que ajusta cuánta aleatoriedad usa el modelo al elegir la siguiente palabra

    # + temp mas baja (0 - 0.3) mas presiso, logico y reptible
    # + temp media (0.5 - 0.8) es un equilibrio entre coherencia y creatividad
    # + temp alta (1.0 - 2.0) mayor creatividad, impredisible e incluso caotico

    GenParams.TEMPERATURE: 0.1, ## Para un bot de QA sobre documentos, queremos precisión (cerca de 0).

    #Penalizacion por repaticion de palabras
    GenParams.REPETITION_PENALTY: 1.1
    }

  # Definicion del modelo (ID)
  modelo_id = 'ibm/granite-3-2-8b-instruct'

  model_id = modelo_id
  parameters = parametros
  project_id = "skills-network"
  watsonx_llm = WatsonxLLM(
      model_id=model_id,
      url="https://us-south.ml.cloud.ibm.com",
      project_id=project_id,
      params=parameters,
  )
  return watsonx_llm

print("LLM Watsonx inicializado correctamente.")

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

#Este modelo convierte fragmentes de textos en representaciones vectoriales
def embedding_generator_watsonx():
    """
    Inicializa y devuelve el modelo de embeddings de Watsonx (Slate).
    """
    #Configuracionde parametros para embeddings
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3, #Limita el texto de entrada a 'x' tokens.
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True} #Hace que la API devuelva el texto final usado (útil para ver truncamientos).
    }

    #Creacion del modelo

    embedding_model = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params
    )

    return embedding_model

def vector_database(chunks):
  """
  Toma los fragmentos de texto y el modelo de embeddings, y crea la base de datos vectorial.
  """


  embedding_model = embedding_generator_watsonx()
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

rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

rag_application.launch(server_name="0.0.0.0", server_port=7869)