import openai
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

def load_docs(directory):
	loader = PyPDFDirectoryLoader(directory)
	documents = loader.load()
	return documents

directory = 'thesis/'
documents = load_docs(directory)

def split_docs(documents, chunk_size=500, chunk_overlap=20):
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
	docs = text_splitter.split_documents(documents)
	return docs

docs = split_docs(documents)
print("number of chuncks: " + str(len(docs)))

which_model = "openai"
#which_model = "huggingface"

features = 0
if(which_model == "openai"):
	features = 1536
if(which_model == "huggingface"):
	features = 384

embeddings = None
if(which_model == "openai"):
	embeddings = OpenAIEmbeddings()
if(which_model == "huggingface"):
	embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

pinecone.init(
	api_key="XXXX",
	environment="gcp-starter"
)

index_name = "mcq-creator"

if index_name not in pinecone.list_indexes():
	pinecone.create_index(
		name=index_name,
		metric='cosine',
		dimension= features
	)
	index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
else:
	index = Pinecone.from_existing_index(index_name, embeddings)

def get_similiar_docs(query, k):
	similar_docs = index.similarity_search(query, k=k)
	return similar_docs

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub

llm = None
if(which_model == "openai"):
	llm = OpenAI(model_name="text-davinci-003",temperature=0)
if(which_model == "huggingface"):
	llm=HuggingFaceHub(repo_id="bigscience/bloom")

chain = load_qa_chain(llm, chain_type="stuff")

def query_agent(query):
	relevant_docs = get_similiar_docs(query, k=15)
	for doc in relevant_docs:
		print(" ")
		print("relevant doc: " + str(doc))
	response = chain.run(input_documents=relevant_docs, question=query)
	return response


st.header("Query your internal documents")
query = st.text_area("Ask me a question about the attached documents")
button = st.button("Submit")

if button:
	response =  query_agent(query)
	st.write(response)
