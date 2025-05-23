import os
import sys
import pickle
import faiss
import readline
from dotenv import load_dotenv
from typing import List

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain

class chatbot:
    # controls how much text each chunk holds and how much they overlap
    chunk_size = 500
    chunk_overlap = 50

    current_dir = os.path.dirname(os.path.abspath(__file__))
    index_file = os.path.join(current_dir, "vectors.faiss")
    store_file = os.path.join(current_dir, "chunks.pkl")
    data_folder = os.path.join(os.path.dirname(current_dir), "data")

    def __init__(self):
        load_dotenv()

        self.openai_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            print("Missing API key. Please set OPENAI_API_KEY in your .env file.")
            sys.exit(1)

        # sets up models needed for embedding and chat
        self.embedding_model = OpenAIEmbeddings(openai_api_key=self.openai_key)
        self.llm = ChatOpenAI(temperature=0, openai_api_key=self.openai_key)

        # holds session data
        self.db = None
        self.chat_history = []

    def prompt_for_reset(self) -> bool:
        while True:
            response = input("Would you like to rebuild the system from scratch? (y/n): ").strip().lower()
            if response in {"y", "n"}:
                return response == "y"
            print("Please respond with 'y' or 'n'.")

    def load_documents(self) -> List[Document]:
        # ensures that user has placed data in the expected folder
        if not os.path.isdir(self.data_folder):
            print(f"The folder '{self.data_folder}' does not exist. Please create it and add your text files.")
            sys.exit(1)

        # filters for .txt files to avoid parsing issues
        files = [f for f in os.listdir(self.data_folder) if f.endswith(".txt")]
        if not files:
            print("No .txt files found. Please add them to the 'data' folder and restart.")
            sys.exit(1)

        docs = []
        for name in files:
            path = os.path.join(self.data_folder, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    docs.append(Document(page_content=content))
            except Exception as e:
                print(f"Failed to read {name}: {e}")

        # chunks the documents to ensure token limits are respected later
        splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = splitter.split_documents(docs)
        print(f"{len(files)} files loaded and processed into smaller parts.")
        return chunks

    def prepare_index(self, chunks: List[Document]) -> FAISS:
        # decides whether to reload or rebuild based on user input and file existence
        reset = self.prompt_for_reset()
        existing = os.path.exists(self.index_file) and os.path.exists(self.store_file)

        if reset or not existing:
            print("Building a new system. May take a second...")

            # builds new index from scratch
            db = FAISS.from_documents(chunks, self.embedding_model)

            # saves vector index and docstore to disk
            faiss.write_index(db.index, self.index_file)
            with open(self.store_file, "wb") as f:
                pickle.dump(db.docstore._dict, f)

            print("The system is ready to use.")
        else:
            print("Loading the previously saved system...")

            # restores previously saved state
            index = faiss.read_index(self.index_file)
            with open(self.store_file, "rb") as f:
                store = pickle.load(f)

            db = FAISS(self.embedding_model.embed_query, index, store, {})
            print("The system has been loaded successfully.")

        return db

    def run(self):
        print("\nWelcome! Lets talk with your files.")

        # loads and prepares data
        chunks = self.load_documents()
        self.db = self.prepare_index(chunks)

        # initializes the retrieval-based chatbot
        qa_chain = ConversationalRetrievalChain.from_llm(self.llm, self.db.as_retriever())

        print("\nAsk your questions below. Type 'exit' to end the session.\n")

        while True:
            try:
                query = input("You: ").strip()
                if query.lower() in {"exit"}:
                    print("Goodbye!")
                    break
                if not query:
                    continue

                # gets a response from the chatbot
                response = qa_chain.invoke({
                    "question": query,
                    "chat_history": [(q, a) for q, a in self.chat_history]
                })

                # shows response and stores interaction history
                print("Assistant:", response["answer"])
                self.chat_history.append((query, response["answer"]))

            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    chatbot().run()