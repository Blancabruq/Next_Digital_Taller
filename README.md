# Agente Experto en Biomecánica

Chatbot inteligente con IA que responde preguntas sobre **sensores inerciales, biomecánica y seguimiento del movimiento articular**. Busca primero en un PDF local y, si no encuentra respuesta, recurre a Internet.

---

## ¿Cómo funciona?

1. El usuario hace una pregunta en el chat.
2. Un guardarraíl comprueba que la pregunta sea relevante para el tema.
3. El agente busca primero en el **documento PDF local** (RAG con ChromaDB).
4. Si no encuentra información, busca en **Internet** (DuckDuckGo).
5. Responde indicando siempre la fuente utilizada.

---

## Requisitos previos

- Python 3.10 o superior
- Una API Key de OpenAI

---

## Instalación y ejecución

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Crear el fichero `.env`

Crea un archivo llamado `.env` en la raíz del proyecto con tu API Key:

```
OPENAI_API_KEY=sk-...
```

### 3. (Solo primera vez) Generar la base de datos vectorial

> Si ya existe la carpeta `chroma_db/`, omite este paso.

```bash
python ingestion.py
```

Esto procesa el PDF de `data/` y crea la base de datos local.

### 4. Lanzar la aplicación

```bash
streamlit run app.py
```

Se abrirá automáticamente el navegador en `http://localhost:8501`.

---

## Estructura del proyecto

```
├── app.py              # Aplicación principal (interfaz + agente)
├── ingestion.py        # Script para procesar el PDF y crear ChromaDB
├── requirements.txt    # Dependencias
├── .env                # API Key (no se sube al repositorio)
├── data/               # Carpeta con el PDF fuente
└── chroma_db/          # Base de datos vectorial generada (no se sube al repositorio)
```
