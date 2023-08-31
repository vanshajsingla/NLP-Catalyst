from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables from .env file
load_dotenv(".env")

# Get environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

# Check if any of the environment variables are missing
if not OPENAI_API_KEY or not OPENAI_API_TYPE or not OPENAI_API_BASE or not OPENAI_API_VERSION:
    st.error("Please provide all the required environment variables.")
    st.stop()

###### GLOBALS
docsDir = ''
qa = ''

####### FUNCTIONS
def directoryLoader(path):
    st.write("Loading documents...")
    # Documentation: https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/directory_loader.html
    loader = DirectoryLoader(path)
    documents = loader.load()
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    from langchain.embeddings import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    from langchain.vectorstores import Chroma
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(engine="gpt-demo",temperature=0), chain_type="stuff", retriever=retriever)
    return qa

def init():
    global docsDir
    docsDir = 'docs/'
    global qa
    qa = directoryLoader(docsDir)

def askQuestion(query):
    return qa.run(query)

# Initialize the app
init()

# Streamlit app
st.title("MultiDoc Magician (Your personalized custom LLM Chatbot 🙂")

with st.chat_message("user"):
    st.write("Hello 👋")
# Get user input
query = st.text_input("Enter your question:")

# Process the question and display the answer
if st.button("Ask"):
    if query:
        answer = askQuestion(query)
        st.write("Answer:", answer)
    else:
        st.warning("Please enter a question.")

