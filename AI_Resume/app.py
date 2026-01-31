import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import pyttsx3
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Load API keys
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM and embeddings
llm = ChatGroq(groq_api_key=groq_api_key, model="openai/gpt-oss-120b")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Text-to-Speech function using pyttsx3 (cross-platform)
def speak_text(text):
    """Convert text to speech"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume level
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        st.error(f"Speech error: {e}")
        return False

# Streamlit UI
st.title("🎓 AI Resume Builder: PDF Chat with Voice")
st.write("Upload PDFs, search content, and chat with AI assistance - now with voice output!")

# Sidebar for PDF upload and settings
with st.sidebar:
    st.header("📁 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        accept_multiple_files=False
    )
    
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            documents = []
            temppdf = "./temp.pdf"
            
            with open(temppdf, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
            
            os.remove(temppdf)
            
            # Split and create embeddings
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=5000, 
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            st.session_state.vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings
            )
            
            st.success(f"✅ Processed {len(splits)} chunks from PDF!")
    
    st.divider()
    
    # Voice settings
    st.header("🔊 Voice Settings")
    enable_voice = st.checkbox("Enable voice output", value=False)
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = ChatMessageHistory()
        st.rerun()

# Main chat interface
st.header("💬 Chat Interface")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your study materials..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    if st.session_state.vectorstore:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = st.session_state.vectorstore.as_retriever()
                
                # Create contextualize question prompt
                contextualize_q_system_prompt = """Given a chat history and the latest user question \
                which might reference context in the chat history, formulate a standalone question \
                which can be understood without the chat history. Do NOT answer the question, \
                just reformulate it if needed and otherwise return it as is."""
                
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                
                history_aware_retriever = create_history_aware_retriever(
                    llm, retriever, contextualize_q_prompt
                )
                
                # Create QA prompt
                qa_system_prompt = """
                Act as a resume builder and help to create a professional resume by 
                gathering information about user's experience,
                skills and education
                
                {context}"""
                
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(
                    history_aware_retriever, 
                    question_answer_chain
                )
                
                # Get response
                response = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history.messages
                })
                
                answer = response["answer"]
                
                st.markdown(answer)
                
                # Add to chat history
                st.session_state.chat_history.add_user_message(prompt)
                st.session_state.chat_history.add_ai_message(answer)
                
                # Text-to-speech if enabled
                if enable_voice:
                    with st.spinner("🔊 Speaking..."):
                        speak_text(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    
    else:
        with st.chat_message("assistant"):
            no_pdf_msg = "Please upload a PDF document first to start chatting!"
            st.warning(no_pdf_msg)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": no_pdf_msg
            })

# Footer
st.divider()
st.caption("💡 Tip: Upload study materials and ask questions to create summaries, get explanations, or test your knowledge!")