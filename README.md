# AI Resume Builder - Conversational RAG Application

## 🎯 Project Overview

AI Resume Builder is an intelligent Streamlit application that helps users create professional resumes through conversational AI. The app uses Retrieval-Augmented Generation (RAG) to analyze uploaded documents and assist users in building comprehensive, well-structured resumes by gathering information about their experience, skills, and education.

## ✨ Key Features

### 1. **PDF Document Processing**
- Upload resume templates, job descriptions, or reference documents
- Automatic text extraction and chunking
- Vector embeddings for semantic search

### 2. **Intelligent Search**
- Quick search through uploaded documents
- Find relevant content instantly
- View top 5 most relevant sections

### 3. **Conversational AI Assistant**
- Chat-based interface for resume building
- Context-aware responses using chat history
- Powered by GROQ's LLM (openai/gpt-oss-120b)

### 4. **Text-to-Speech Output**
- Toggle voice output for AI responses
- Cross-platform support (Windows, macOS, Linux)
- Adjustable speech rate and volume

### 5. **Chat History Management**
- Maintains conversation context
- Clear chat history option
- Persistent session state

## 🛠️ Technology Stack

### Core Technologies
- **Streamlit**: Web application framework
- **LangChain**: LLM orchestration and RAG pipeline
- **GROQ API**: Large Language Model inference
- **HuggingFace**: Embeddings (all-MiniLM-L6-v2)
- **ChromaDB**: Vector database for document storage

### Key Libraries
- **langchain-groq**: GROQ LLM integration
- **langchain-huggingface**: HuggingFace embeddings
- **langchain-community**: Document loaders and vector stores
- **pypdf**: PDF processing
- **pyttsx3**: Text-to-speech conversion
- **python-dotenv**: Environment variable management


## 📁 Project Structure

```
ai-resume-builder/
│
├── enhanced_rag_app.py      # Main application file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .env                      # Environment variables (create this)
└── temp.pdf                  # Temporary file (auto-created/deleted)
```

## 🔧 Configuration

### LLM Model
Currently using: `openai/gpt-oss-120b`

To change the model, modify this line in the code:
```python
llm = ChatGroq(groq_api_key=groq_api_key, model="openai/gpt-oss-120b")
```

### Embedding Model
Currently using: `all-MiniLM-L6-v2`

To change the embedding model:
```python
embeddings = HuggingFaceEmbeddings(model_name="your-preferred-model")
```

### Text Chunking
Adjust chunk size and overlap:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,      # Adjust based on document complexity
    chunk_overlap=200     # Overlap for context preservation
)
```

## 🎯 Use Cases

1. **Resume Creation**: Build professional resumes from scratch
2. **Resume Refinement**: Improve existing resume content
3. **Job Description Analysis**: Analyze JDs and tailor resume accordingly
4. **Skill Articulation**: Get help describing technical and soft skills
5. **Experience Formatting**: Professional formatting for work experience
6. **Achievement Quantification**: Help quantify achievements with metrics


### RAG Pipeline
1. **Document Loading**: PDF files are loaded using PyPDFLoader
2. **Text Splitting**: Documents split into manageable chunks
3. **Embedding**: Text chunks converted to vector embeddings
4. **Vector Storage**: Embeddings stored in ChromaDB
5. **Retrieval**: Relevant chunks retrieved based on user queries
6. **Generation**: LLM generates responses using retrieved context
