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

## 📋 Prerequisites

- Python 3.8 or higher
- GROQ API key ([Get it here](https://console.groq.com))
- HuggingFace token ([Get it here](https://huggingface.co/settings/tokens))

## 🚀 Installation

### 1. Clone or Download the Project
```bash
# If using git
git clone <your-repo-url>
cd ai-resume-builder

# Or download and extract the files
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the project root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

**How to get API keys:**
- **GROQ API Key**: Sign up at [console.groq.com](https://console.groq.com) and generate an API key
- **HuggingFace Token**: Create an account at [huggingface.co](https://huggingface.co) and generate a token from Settings > Access Tokens

## 💻 Usage

### 1. Start the Application
```bash
streamlit run enhanced_rag_app.py
```

The app will open in your default browser at `http://localhost:8501`

### 2. Upload Documents
- Click on "Choose a PDF file" in the sidebar
- Upload resume templates, job descriptions, or reference documents
- Wait for processing confirmation

### 3. Search Documents (Optional)
- Expand the "Search Document" section
- Enter keywords to find relevant content
- Review the top 5 matching sections

### 4. Chat with AI
- Type your questions in the chat input
- Ask about resume formatting, skill descriptions, experience wording
- Request help with specific sections

### 5. Enable Voice Output (Optional)
- Toggle "Enable voice output" in the sidebar
- AI responses will be read aloud

### 6. Build Your Resume
Example conversation flow:
```
User: Help me describe my software engineering experience
AI: I'd be happy to help! Can you tell me about your role, key responsibilities, and major achievements?

User: I worked as a backend developer for 2 years, mainly using Python and Django
AI: Great! Here's a professional way to describe that experience:
- Developed and maintained scalable backend services using Python and Django...
```

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

## 🔍 How It Works

### RAG Pipeline
1. **Document Loading**: PDF files are loaded using PyPDFLoader
2. **Text Splitting**: Documents split into manageable chunks
3. **Embedding**: Text chunks converted to vector embeddings
4. **Vector Storage**: Embeddings stored in ChromaDB
5. **Retrieval**: Relevant chunks retrieved based on user queries
6. **Generation**: LLM generates responses using retrieved context

### Chat History
- Maintains conversation context using `ChatMessageHistory`
- Creates history-aware retriever for contextual responses
- Reformulates questions based on chat history

## ⚠️ Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
pip install --upgrade -r requirements.txt
```

**2. API Key Errors**
- Verify `.env` file exists in project root
- Check API keys are valid and have proper permissions
- Ensure no extra spaces in `.env` file

**3. Text-to-Speech Not Working**
- On Linux: Install espeak (`sudo apt-get install espeak`)
- On macOS: Should work out of the box
- On Windows: Ensure SAPI5 is available

**4. ChromaDB Issues**
- Clear the database: Delete any `chroma_db` folders
- Restart the application

## 🔒 Privacy & Security

- **API Keys**: Never commit `.env` file to version control
- **Documents**: Uploaded PDFs are processed temporarily and deleted
- **Data**: Vector embeddings stored locally, not sent to external services
- **Sessions**: Chat history cleared when browser session ends

## 🚀 Future Enhancements

Potential features for future versions:
- [ ] Export resume to PDF/DOCX format
- [ ] Multiple resume templates
- [ ] ATS (Applicant Tracking System) optimization
- [ ] LinkedIn profile integration
- [ ] Cover letter generation
- [ ] Multi-language support
- [ ] Resume version comparison

## 📝 License

This project is open source and available for educational and personal use.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📧 Support

For questions or issues:
1. Check the troubleshooting section
2. Review GROQ and LangChain documentation
3. Open an issue in the repository

## 🙏 Acknowledgments

- **GROQ**: For providing fast LLM inference
- **LangChain**: For the RAG framework
- **HuggingFace**: For embedding models
- **Streamlit**: For the web framework

---

**Built with ❤️ for helping people create better resumes**

**Version**: 1.0.0  
**Last Updated**: January 2026