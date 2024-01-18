import streamlit as st 
import os
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch 
import textwrap 
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA 
from constants import CHROMA_SETTINGS
from streamlit_chat import message
from docx import Document  # Add this import for working with DOCX files
import requests
from bs4 import BeautifulSoup
import tempfile
import shutil

st.set_page_config(layout="wide")

device = torch.device('cpu')
col1, col2 = st.columns([1, 2])
checkpoint = "MBZUAI/LaMini-T5-738M"
print(f"Checkpoint path: {checkpoint}")  # Add this line for debugging
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map=device,
    torch_dtype=torch.float32
)

persist_directory = "db"

def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def read_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text
    else:
        st.error(f"Failed to fetch content from URL. Status code: {response.status_code}")
        return None

@st.cache_resource
def data_ingestion(file_path_or_url):
    if file_path_or_url.endswith(".pdf"):
        print(file_path_or_url)
        loader = PDFMinerLoader(file_path_or_url)
    elif file_path_or_url.endswith(".docx"):
        print(file_path_or_url)
        docx_text = read_docx(file_path_or_url)
        # Create a temporary text file for each DOCX document
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "dummy.txt")
        with open(temp_file_path, "w", encoding="utf-8") as temp_file:
            temp_file.write(docx_text)
        loader = DirectoryLoader("docs")
    elif file_path_or_url.endswith(".txt"):
        print(file_path_or_url)
        txt_text = read_txt(file_path_or_url)
        # Create a temporary text file for each TXT document
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "dummy.txt")
        with open(temp_file_path, "w", encoding="utf-8") as temp_file:
            temp_file.write(txt_text)
        loader = DirectoryLoader("docs")
    elif file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
        text = read_url(file_path_or_url)
        if text is None:
            return
        # Create a temporary text file for the URL content
        temp_file_path = "docs/dummy.txt"
        with open(temp_file_path, "w", encoding="utf-8") as temp_file:
            temp_file.write(text)
        loader = DirectoryLoader("docs")
    else:
        # Handle other file types or raise an error as needed
        raise ValueError(f"Unsupported file type: {file_path_or_url}")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store here
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=4000,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        device=device
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

# function to display the PDF of a given file 
@st.cache_data
def displayPDF(filepath):
    with open(filepath, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        # Ensure unique keys for user and generated messages
        user_key = str(i) + "_user"
        generated_key = str(i) + "_generated"

        # Display user message
        message(history["past"][i], is_user=True, key=user_key)

        # Display generated message
        message(history["generated"][i], key=generated_key)

import streamlit as st

def main():
    # Header with blue background
    header_style = "background-color: #3498db; padding: 1rem;"
    st.markdown("<h1 style='text-align: center; color: white;" + header_style + "'>Chat with your PDF, Docx, Txt, or any URL</h1>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;" + header_style + "'>Built by Rajni Sharma. AI Anytime with ❤️ </h3>", unsafe_allow_html=True)

    # Radio button for input selection
    st.markdown("<p style='color: #3498db; font-size: 1.2rem;'>Select Input Option:</p>", unsafe_allow_html=True)

    input_option = st.radio("", ["Upload Document", "Enter URL"], key="input_option", index=0)

    if input_option == "Upload Document":
        # File upload section
        uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])

        if uploaded_file is not None:
            # Display file details and preview in columns
            st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
            st.markdown("<h2 style='color: #3498db;'>Uploaded Document Details</h2>", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
            process_uploaded_file(uploaded_file)

    elif input_option == "Enter URL":
        # URL input section
        user_input_url = st.text_input("Enter a URL:", key="input_url")

        if user_input_url:
            # Display file details and preview in columns
            st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
            st.markdown("<h2 style='color: #3498db;'>URL Details</h2>", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
            process_url_input(user_input_url)

def process_uploaded_file(uploaded_file):
    file_details = get_file_details(uploaded_file)
    filepath = save_uploaded_file(uploaded_file)

    # Display file details in the left column
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("<h3 style='color: #3498db;'>File Details</h3>", unsafe_allow_html=True)
        st.json(file_details)
        st.markdown("<h4 style='color: #3498db;'>File Preview</h4>", unsafe_allow_html=True)
        display_file_preview(filepath)

    # Display chat history in the right column
    with col2:
        st.markdown("<h3 style='color: #3498db;'>Chat History</h3>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
        with st.spinner('Embeddings are in process...'):
            ingested_data = data_ingestion(filepath)
        st.success('Embeddings are created successfully!')

        st.markdown("<h4 style='color: #3498db;'>Chat Here</h4>", unsafe_allow_html=True)
        user_input = st.text_input("", key="input", help="Enter your query")

        # Initialize session state for generated responses and past messages
        if "generated" not in st.session_state:
            st.session_state["generated"] = ["I am ready to help you"]
        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey there!"]

        # Search the database for a response based on user input and update session state
        if user_input:
            answer = process_answer({'query': user_input})
            st.session_state["past"].append(user_input)
            response = answer
            st.session_state["generated"].append(response)

        # Display conversation history using Streamlit messages
        if st.session_state["generated"]:
            st.markdown("<h4 style='color: #3498db;'>Conversation History</h4>", unsafe_allow_html=True)
            display_conversation(st.session_state)

def process_url_input(user_input_url):
    st.info(f"You entered the URL: {user_input_url}")

    with st.spinner('Embeddings are in process...'):
        ingested_data = data_ingestion(user_input_url)
    st.success('Embeddings are created successfully!')

    st.markdown("<h4 style='color: #3498db;'>Chat Here</h4>", unsafe_allow_html=True)
    user_input = st.text_input("", key="input", help="Enter your query")

    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    # Search the database for a response based on user input and update session state
    if user_input:
        answer = process_answer({'query': user_input})
        st.session_state["past"].append(user_input)
        response = answer
        st.session_state["generated"].append(response)

    # Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        st.markdown("<h4 style='color: #3498db;'>Conversation History</h4>", unsafe_allow_html=True)
        display_conversation(st.session_state)

def empty_folder(folder_path):
    """Empty the contents of a folder."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def get_file_details(uploaded_file):
    file_details = {
        "Filename": uploaded_file.name,
        "File size": get_file_size(uploaded_file)
    }
    return file_details
def save_uploaded_file(uploaded_file):
    filepath = "docs/" + uploaded_file.name
    with open(filepath, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    return filepath
def display_file_preview(filepath):
    st.markdown("<h4 style='color:black;'>File preview</h4>", unsafe_allow_html=True)
    if filepath.endswith(".pdf"):
        pdf_view = displayPDF(filepath)
    elif filepath.endswith(".docx"):
        st.text(read_docx(filepath))
    elif filepath.endswith(".txt"):
        st.text(read_txt(filepath))
def delete_and_empty_folders():
    if os.path.exists("docs") and os.path.isdir("docs"):
        empty_folder("docs")
    if os.path.exists("db") and os.path.isdir("db"):
        empty_folder("db")

if __name__ == "__main__":
    main()
