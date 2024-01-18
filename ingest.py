import os
import tempfile
import fitz  # PyMuPDF
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from docx import Document
from constants import CHROMA_SETTINGS

persist_directory = "db"

def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    return text

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def main():
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory
    try:
        texts_with_ids = []  # List to store texts with unique IDs

        for root, dirs, files in os.walk("docs"):
            for i, file in enumerate(files):
                file_path = os.path.join(root, file)
                if file.endswith(".pdf"):
                    print(file)
                    pdf_text = read_pdf(file_path)
                    texts_with_ids.append({"id": f"{i}_pdf", "text": pdf_text})
                elif file.endswith(".docx"):
                    print(file)
                    docx_text = read_docx(file_path)
                    texts_with_ids.append({"id": f"{i}_docx", "text": docx_text})
                elif file.endswith(".txt"):
                    print(file)
                    txt_text = read_txt(file_path)
                    texts_with_ids.append({"id": f"{i}_txt", "text": txt_text})


        # Use DirectoryLoader to load documents from the temporary directory
        loader = DirectoryLoader(temp_dir)
        documents = loader.load()

        print("splitting into chunks")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        # Create embeddings here
        print("Loading sentence transformers model")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create vector store here
        print(f"Creating embeddings. May take some minutes...")
        
        # Ensure that each text has a unique identifier (ID)
        texts_with_ids += [{"id": f"{i}_text", "text": text} for i, text in enumerate(texts)]
        print(texts_with_ids)

        if not texts_with_ids:
            print("No texts to process. Aborting.")
            return

        # Extract 'text' values from the dictionaries in texts_with_ids
        text_list = [item["text"] for item in texts_with_ids]

        db = Chroma.from_texts(text_list, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
        db.persist()
        db = None

        print(f"Ingestion complete! You can now run privateGPT.py to query your documents")
    finally:
        # Cleanup: Remove the temporary directory and its contents
        for file_name in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file_name)
            os.remove(file_path)
        os.rmdir(temp_dir)

if __name__ == "__main__":
    main()
