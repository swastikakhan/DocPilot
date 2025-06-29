import os
import shutil
import openai
import nltk

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ---------------------------------------------
# 🔧 Patch for NLTK "punkt_tab" error in unstructured
# ---------------------------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Monkey-patch to bypass NLTK's broken punkt_tab fallback
from nltk import tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer

def patched_get_tokenizer(language):
    if language == "english":
        return PunktSentenceTokenizer()
    else:
        raise ValueError(f"Unsupported language: {language}")

tokenize._get_tokenizer = patched_get_tokenizer
# Patch NLTK to redirect "punkt_tab" to "punkt"
import types
_original_find = nltk.data.find
def patched_find(resource_name, *args, **kwargs):
    if resource_name.startswith("tokenizers/punkt_tab"):
        resource_name = resource_name.replace("tokenizers/punkt_tab", "tokenizers/punkt", 1)
    return _original_find(resource_name, *args, **kwargs)
nltk.data.find = patched_find
# --- End patch ---
# ---------------------------------------------
# 🔐 Load environment variables and API key
# ---------------------------------------------
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

# ---------------------------------------------
# 📂 Paths
# ---------------------------------------------
CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

# ---------------------------------------------
# 🚀 Main RAG setup pipeline
# ---------------------------------------------
def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split {len(documents)} documents into {len(chunks)} chunks.")

    # Debug sample
    document = chunks[10]
    print("📄 Sample chunk content:\n", document.page_content[:300], "\n")
    print("📄 Metadata:\n", document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"✅ Saved {len(chunks)} chunks to {CHROMA_PATH}.")

# ---------------------------------------------
# 🔃 Entry point
# ---------------------------------------------
if __name__ == "__main__":
    main()
