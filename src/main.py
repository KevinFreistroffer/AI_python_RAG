# Environment setup - Load variables from .env file for secure API key storage
import os
from dotenv import load_dotenv
load_dotenv()

# API Keys for various services
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# Import required libraries
# Pinecone - A vector database service that stores and searches high-dimensional vectors
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeVectorStore

# Document Loading - Tools to read and process text files
from langchain_community.document_loaders import TextLoader

# Embedding Models - Convert text into numerical vectors that capture meaning
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint

# Language Models (LLMs) - AI models that understand and generate human-like text
from langchain_community.llms import HuggingFaceHub


# QA Chain - Combines components to create a question-answering system
from langchain.chains import RetrievalQA

# Initialize Embedding Model
# This model converts text into 384-dimensional vectors that represent the meaning of the text
# Smaller, faster model suitable for basic text understanding
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Language Model
# FLAN-T5 is a general-purpose language model that can understand and generate text
# 'small' version is lighter and faster, but less capable than larger models
# llm = HuggingFaceHub(
#     repo_id="google/flan-t5-small",
#     huggingfacehub_api_token=HUGGING_FACE_API_KEY,
#     task="text2text-generation",
#     model_kwargs={
#         "temperature": 0.5,
#         "max_length": 512
#     }
# )

llm = HuggingFaceEndpoint(
    task='text-generation',
    model="deepseek-ai/DeepSeek-R1",
    max_new_tokens=100,
    temperature=0.7,
    huggingfacehub_api_token=HUGGING_FACE_API_KEY
)

# Load and prepare documents
# TextLoader reads the content of text files into a format that LangChain can process
loader = TextLoader("src/data.txt")
documents = loader.load()

# Initialize Pinecone
# Pinecone is a vector database that allows efficient storage and similarity search of vectors
pc = Pinecone(api_key=PINECONE_API_KEY)

# Setup Pinecone Index
# An index is like a database table that stores vectors and allows searching them
# Each index needs to be configured for the specific size of vectors we're using
index_name = "starter-index"
indexes = pc.list_indexes()

# Create a new index if it doesn't exist
if not any(index["name"] == index_name for index in indexes):
    pc.create_index(
        index_name,
        dimension=384,  # Changed from 1536 to 384 to match the embedding model
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Serverless deployment on AWS
    )

index = pc.Index(index_name)

# Create Vector Store
# This process:
# 1. Takes our documents
# 2. Converts them to vectors using the embedding model
# 3. Stores these vectors in Pinecone for quick similarity search
vectorstore = PineconeVectorStore.from_documents(
    documents,
    embedding_model,
    index_name=index_name
)

# Setup QA Chain
# This creates a pipeline that:
# 1. Takes a question
# 2. Converts it to a vector
# 3. Finds similar vectors (and their associated text) in Pinecone
# 4. Sends the question and relevant text to the LLM
# 5. Returns the LLM's answer
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" means we send all relevant context to the LLM at once
    retriever=vectorstore.as_retriever()
)

# Run Query
# Example question about lunar eclipse
query = "What is the date of the first lunar eclipse in 2024?"
response = qa_chain.invoke(query)
print(response)



