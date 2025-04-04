import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class RAGAgent:
    def __init__(self, topic):
        """
        Initializes the RAGAgent for the specified topic.

        Args:
            topic (str): The detected topic to select the appropriate vectorstore.
        """
        self.vectorstore_path = self.get_vectorstore_path(topic)
        self.vectorstore = self.load_vectorstore()

    def get_vectorstore_path(self, topic):
        return os.path.join(r"data\embeddings",topic)
    
    def load_vectorstore(self):

        faiss_file = os.path.join(self.vectorstore_path, "index.faiss")
        pkl_file = os.path.join(self.vectorstore_path, "index.pkl")

        try:
            if os.path.exists(faiss_file) and os.path.exists(pkl_file):
                print(f"Loading vectorstore for topic: {self.vectorstore_path}")
                vectorstore = FAISS.load_local(
                    self.vectorstore_path, 
                    OpenAIEmbeddings(), 
                    allow_dangerous_deserialization=True
                )
                print("Vectorstore loaded successfully.")
                return vectorstore
            else:
                print(f"Missing vectorstore files in {self.vectorstore_path}. Returning an empty vectorstore.")
                embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
                empty_vectorstore = FAISS(embeddings)
                return empty_vectorstore
        except Exception as e:
            print(f"An unexpected error occurred while loading the vectorstore: {e}")
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            empty_vectorstore = FAISS(embeddings)
            return empty_vectorstore

    def handle_query(self, user_query):
        try:
            retriever_info = ""
            retriever_output = self.vectorstore.as_retriever(search_kwargs={"k": 15}).get_relevant_documents(user_query)
            for doc in retriever_output:
                print(f"- {doc.page_content}")
                retriever_info = retriever_info + doc.page_content
            return retriever_info

        except Exception as e:
            print(f"Error during query handling: {e}")
            return f"Error: {e}"

