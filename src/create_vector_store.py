import os
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def read_pdf_text(pdf_path, timeout=5):
    """Reads and extracts text from a PDF file with a timeout per page."""
    import concurrent.futures
    text = ""
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                future = executor.submit(page.extract_text)
                page_text = future.result(timeout=timeout)
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def get_text_chunks(text):
    """Splits extracted text into smaller chunks for embedding."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def create_vectorstore_for_topic(topic, topic_folder_path, vectorstore_path):
    """Creates and saves a FAISS vectorstore for a given topic folder."""
    if not os.path.exists(topic_folder_path):
        print(f"Folder for topic '{topic}' not found at {topic_folder_path}. Skipping...")
        return
    
    pdf_files = [
        os.path.join(topic_folder_path, f)
        for f in os.listdir(topic_folder_path)
        if f.endswith(".pdf")
    ]

    if not pdf_files:
        print(f"No PDF files found in topic folder '{topic_folder_path}'. Skipping...")
        return

    if os.path.exists(vectorstore_path + '.faiss'):
        print(f"Loading existing vectorstore for topic '{topic}' from disk.")
        return FAISS.load_local(vectorstore_path, OpenAIEmbeddings())

    print(f"Creating new vectorstore for topic '{topic}'.")
    raw_text = "".join([read_pdf_text(pdf) for pdf in pdf_files])
    text_chunks = get_text_chunks(raw_text)
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        disallowed_special=()
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local(vectorstore_path)
    print(f"FAISS vectorstore saved at '{vectorstore_path}'.")
    return vectorstore

def main():
    topics = ["side_channel_attack","Information_Leakage"]
    
    topic_base_directory = r"data\papers"
    
    output_directory = r"data\embeddings"
    os.makedirs(output_directory, exist_ok=True)

    for topic in topics:
        topic_folder_path = os.path.join(topic_base_directory, topic)
        vectorstore_path = os.path.join(output_directory, topic)
        create_vectorstore_for_topic(topic, topic_folder_path, vectorstore_path)
    
    print("Vectorstore creation complete.")

if __name__ == "__main__":
    main()
