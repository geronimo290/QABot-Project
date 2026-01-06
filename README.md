# 🤖 QABot: Asistente Documental con Llama 3.3 & RAG

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Open%20Live%20Demo-blue)](https://huggingface.co/spaces/[TU_USUARIO]/[NOMBRE_DEL_SPACE])
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?style=flat)

## 📖 Descripción del Proyecto

**QABot** es una herramienta de Inteligencia Artificial diseñada para "conversar" con documentos PDF. Utiliza una arquitectura **RAG (Retrieval-Augmented Generation)** moderna y 100% Open Source.

Este proyecto representa una **migración técnica** desde una solución propietaria (IBM Watsonx) hacia un stack flexible y gratuito.

### 🚀 Características Principales
* **Análisis de Documentos:** Carga PDFs y extrae información clave al instante.
* **Motor LLM Avanzado:** Utiliza `llama-3.3-70b-versatile` para respuestas precisas.
* **Privacidad Local:** Embeddings generados localmente con `sentence-transformers`.

---

## 🆚 Comparativa: Migración Tecnológica

Este proyecto comenzó como una implementación en IBM Cloud y evolucionó a una solución Open Source.

| Característica | V1 (Legacy - IBM) | V2 (Actual - Open Source) |
| :--- | :--- | :--- |
| **Modelo** | IBM Granite-3-8b | **Llama-3.3-70b** |
| **Costo** | Créditos Cloud | **Gratuito** (Groq API) |
| **Infraestructura** | Vendor Lock-in | **Agnóstica / Local** |

---

## 🛠️ Stack Tecnológico

* **Orquestación:** [LangChain](https://www.langchain.com/)
* **LLM:** Meta Llama 3.3 (vía Groq)
* **Vector DB:** ChromaDB
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Frontend:** Gradio

---

## ⚙️ Instalación Local

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/](https://github.com/)[TU_USUARIO]/[TU_REPO].git
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configurar API Key:**
    Renombra el archivo `.env.example` a `.env` y agrega tu clave de Groq:
    ```bash
    GROQ_API_KEY="gsk_..."
    ```

4.  **Ejecutar:**
    ```bash
    python app.py
    ```

---

## 👤 Autor

**Gerónimo Pautazzo**
* [LinkedIn]([https://www.linkedin.com/in/gero-pautazzo-88900325a/])
